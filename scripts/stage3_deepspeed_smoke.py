"""Eight-rank tests of LCLM's separate DeepSpeed backward/update path."""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin

from scripts.stage3_nccl_smoke import Harness
from train.deepspeed_step import COMPRESSION_GROUP, DeepSpeedStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--out', required=True)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    config = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 2,
              'gradient_clipping': 0.0 if args.linear else 1.0, 'bf16': {'enabled': True},
              'zero_optimization': {'stage': args.stage, 'overlap_comm': False,
                                    'reduce_bucket_size': 100000, 'allgather_bucket_size': 100000},
              'steps_per_print': 100000, 'zero_allow_untested_optimizer': True}
    if args.offload:
        config['zero_optimization']['offload_optimizer'] = {'device': 'cpu', 'pin_memory': True}
    if args.fp16:
        config['bf16']['enabled'] = False
        config['fp16'] = {'enabled': True, 'loss_scale': 0, 'initial_scale_power': 4,
                          'loss_scale_window': 1000, 'hysteresis': 1}
    accelerator = Accelerator(gradient_accumulation_steps=2,
                              deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=config))
    torch.manual_seed(1234)
    if args.linear:
        # Analytic gradient-accumulation check: (2 + 6) / 2 = 4, so a
        # 0.125 SGD update moves every weight from 1 to exactly 0.5.
        class LinearLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(16))
            def forward(self, coefficient):
                return (self.weight * coefficient).sum()
        model = LinearLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
        model, optimizer = accelerator.prepare(model, optimizer)
        controller = DeepSpeedStep(model, accelerator)
        for coefficient in (2.0, 6.0):
            with accelerator.accumulate(model):
                controller.begin_microbatch()
                assert controller.backward_and_step(model(coefficient), local_has_memory=True)
        masters = [p for group in model.basic_optimizer.param_groups for p in group['params']]
        assert all(torch.equal(p, torch.full_like(p, 0.5)) for p in masters)
        if rank == 0:
            print(f'PASS ANALYTIC ACCUMULATION ZeRO-{args.stage}', flush=True)
        accelerator.end_training()
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    model = Harness()
    optimizer = torch.optim.AdamW([
        {'params': model.decoder.parameters(), COMPRESSION_GROUP: False},
        {'params': model.core.encoder.parameters(), COMPRESSION_GROUP: True},
        {'params': model.core.adapter.parameters(), COMPRESSION_GROUP: True}], lr=1e-3, weight_decay=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    controller = DeepSpeedStep(model, accelerator)
    base = model.basic_optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(base, step_size=1, gamma=0.95)

    def snapshot():
        # Master parameters/state expose tiny updates hidden by BF16 rounding.
        groups = [(i, p) for i, group in enumerate(base.param_groups) for p in group['params']]
        return { (i, j): {'weight': p.detach().clone(), 'state': copy.deepcopy(base.state.get(p, {}))}
                 for j, (i, p) in enumerate(groups) }

    def equal(left, right):
        if torch.is_tensor(left):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        return left == right

    reports = []
    # First native window before lazy state exists; seed momentum; native after
    # momentum; memory only on rank 0/first microbatch; bad loss with finite
    # gradients; bad gradients alone; clean recovery after each rejection.
    patterns = [('native', None, 0), ('memory', None, 0), ('native', None, 0),
                ('asymmetric', None, 0), ('memory', 'loss', 0), ('memory', None, 0),
                ('memory', 'gradient', 1), ('memory', None, 0)]
    for window, (pattern, bad, bad_micro) in enumerate(patterns):
        before = snapshot()
        scheduler_before = scheduler.last_epoch
        scale_before = model.optimizer.loss_scale if args.fp16 else None
        has_window_memory = False
        for micro in range(2):
            has = pattern == 'memory' or (pattern == 'asymmetric' and rank == 0 and micro == 0)
            has_window_memory |= has
            with accelerator.accumulate(model):
                controller.begin_microbatch()
                loss = model(has)
                # Adding a NaN constant leaves the loss derivatives finite.
                if bad == 'loss' and micro == bad_micro and rank == 0:
                    loss = loss + torch.tensor(float('nan'), device=loss.device)
                handle = None
                if bad == 'gradient' and micro == bad_micro and rank == 0:
                    # Inject a bad derivative while keeping the loss finite.
                    parameter = next(p for p in model.module.decoder.parameters() if p.requires_grad)
                    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
                valid = controller.backward_and_step(loss, local_has_memory=has_window_memory)
                if handle is not None:
                    handle.remove()
                if accelerator.sync_gradients and controller.last_step_applied:
                    scheduler.step()
                if micro == 0:
                    assert equal(before, snapshot()), 'Weights/state changed inside accumulation'
        after = snapshot()
        if bad:
            assert not valid and not controller.last_step_applied
            assert equal(before, after), 'Rejected window changed weights or optimizer state'
            assert scheduler.last_epoch == scheduler_before, 'Rejected window advanced scheduler'
            if args.fp16:
                assert model.optimizer.loss_scale < scale_before, 'Rejected FP16 window did not lower the loss scale'
        else:
            assert valid and controller.last_step_applied
            assert scheduler.last_epoch == scheduler_before + 1
            assert not equal(before, after), 'Valid window failed to update'
            if pattern == 'native':
                assert all(equal(before[k], after[k]) for k in before if k[0] in (1, 2)), 'Unused compression weights/state moved'
            else:
                assert any(not equal(before[k], after[k]) for k in before if k[0] in (1, 2)), 'Compression did not update'
        assert all(torch.isfinite(record['weight']).all() for record in after.values())
        dist.barrier()
        record = {'window': window, 'pattern': pattern, 'bad': bad, 'passed': True}
        reports.append(record)
        if rank == 0:
            print('PASS DEEPSPEED', args.stage, args.offload, json.dumps(record), flush=True)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    (root / f'zero{args.stage}-offload{int(args.offload)}-fp16{int(args.fp16)}-rank{rank}.json').write_text(json.dumps(reports, indent=2))
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
