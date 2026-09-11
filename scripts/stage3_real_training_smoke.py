"""Run real packed 16k/32k batches through the production LCLMTrainer on 8 GPUs."""
import copy
import argparse
import json
import os
import pickle
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from train.launch_train import ModelArguments, DataArguments, TrainingConfig
from train.trainer import LCLMTrainer
from data.packing_utils import collate_packed_batch

ROOT = Path('/data/stage3-build-20260906/readiness-20260910-v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['fsdp', 'deepspeed'], default='fsdp')
    parser.add_argument('--report-root', default=str(ROOT))
    args = parser.parse_args()
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    model_args = ModelArguments(
        decoder_name=str(ROOT / 'models/decoder'), decoder_tokenizer_name=str(ROOT / 'models/decoder'),
        embed_model_name=str(ROOT / 'models/encoder'), train_decoder=True, train_encoder=True,
        decoder_attn_implementation='flash_attention_2', embed_attn_implementation='flash_attention_2',
        encoder_window_size=1024, max_encode_batch_size=32, encoder_mask_type='bidirectional',
        use_fused_ce=True)
    data_args = DataArguments(dataset_name=str(ROOT / 'fixture-16384'), train_batch_size=1,
                              max_packed_length=16384, packed_attention_backend='flash')
    train_args = TrainingConfig(
        compression_ratio=16, distributed_type=args.backend, use_liger_kernel=False,
        train_wrap_tokens=True, gradient_accumulation_steps=2, max_steps=8, num_epochs=4,
        decoder_lr=1e-5, encoder_lr=1e-5, adapter_lr=1e-5,
        decoder_betas='0.9,0.95', encoder_betas='0.9,0.95', adapter_betas='0.9,0.95',
        decoder_weight_decay=0.01, encoder_weight_decay=0.01, adapter_weight_decay=0.01,
        warmup_steps=0, auto_resume=False, output_dir=str(report_root / 'training-smoke-output'),
        wandb_project='stage3-readiness', save_steps=0, seed=4231,
        decoder_gradient_checkpointing=True, encoder_gradient_checkpointing=True)
    trainer = LCLMTrainer(model_args, data_args, train_args)
    assert trainer.accelerator.num_processes == 8
    assert trainer.accelerator.distributed_type.name == args.backend.upper()
    trainer.model.train()
    dataset = trainer.train_dataloader.dataset
    # Verify StatefulDataLoader exact resume on real expanded inputs.
    iterator = iter(trainer.train_dataloader)
    next(iterator)
    saved = copy.deepcopy(trainer.train_dataloader.state_dict())
    expected = next(iterator)
    trainer.train_dataloader.load_state_dict(saved)
    actual = next(iter(trainer.train_dataloader))
    assert torch.equal(expected['input_ids'], actual['input_ids'])
    assert torch.equal(expected['labels'], actual['labels'])
    assert expected['memory_token_ids'] == actual['memory_token_ids']
    report = {'rank': rank, 'loader_resume_exact': True, 'steps': []}
    micro_step = 0
    for length in (16384, 32768):
        table = pq.read_table(ROOT / f'fixture-{length}/samples.parquet').to_pylist()
        samples = {}
        for row in table:
            samples.setdefault(row['sample_kind'], []).append(row)
        dataset.target_length = length
        # Four optimizer steps: all native; asymmetric memory; all memory;
        # all native after AdamW momentum has been established.
        kinds = ['native', 'native', 'expansion' if rank == 0 else 'native',
                 'reasoning' if rank % 2 else 'other', 'expansion', 'other', 'native', 'native']
        for local_step, kind in enumerate(kinds):
            row = samples[kind][rank % len(samples[kind])]
            examples = pickle.loads(row['packed_batch_bytes'])
            expanded = [dataset._expand_example(example) for example in examples]
            assert all(example is not None for example in expanded)
            assert sum(example['seq_len'] for example in expanded) == row['expanded_seq_len_cs16']
            batch = collate_packed_batch(expanded, target_length=length,
                                        pad_token_id=trainer.decoder_tokenizer.pad_token_id)
            assert batch['input_ids'].shape == (1, length)
            assert int((batch['labels'] != -100).sum()) > 0
            for start, stop in batch['memory_positions'][0]:
                assert (batch['labels'][0, start - 1:stop + 1] == -100).all()
            batch = trainer._move_batch_to_device(batch)
            def compression_steps():
                if args.backend == 'deepspeed':
                    engine = trainer.model
                    base = engine.basic_optimizer
                    parameters = [p for i in trainer._compression_param_group_indices
                                  for p in base.param_groups[i]['params']]
                    return [float(base.state[p].get('step', 0)) for p in parameters]
                return [float(trainer.optimizer.state[p].get('step', 0))
                        for index in trainer._compression_param_group_indices
                        for p in trainer.optimizer.param_groups[index]['params']]
            before = compression_steps()
            with trainer.accelerator.accumulate(trainer.model):
                loss = trainer._training_step(batch, micro_step)
                assert loss is not None and torch.isfinite(torch.tensor(loss))
                sync = trainer.accelerator.sync_gradients
            after = compression_steps()
            if local_step in (0, 1, 6, 7):
                assert before == after, 'Encoder/adapter optimizer advanced on globally uncompressed step'
            if sync and local_step in (3, 5):
                assert any(b > a for a, b in zip(before, after)), 'Compression optimizer did not advance'
            report['steps'].append({'length': length, 'kind': kind, 'loss': loss,
                                    'optimizer_boundary': sync, 'micro_step': micro_step})
            micro_step += 1
            if sync:
                trainer.global_step += 1
            dist.barrier()
            if rank == 0:
                print(f'PASS REAL TRAIN {length} microbatch={local_step} loss={loss:.6f} optimizer={sync}', flush=True)
    report['max_gpu_bytes'] = torch.cuda.max_memory_allocated()
    (report_root / f'real-training-rank-{rank}.json').write_text(json.dumps(report, indent=2))
    dist.barrier()
    trainer.accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
