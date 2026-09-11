"""Real LCLM encoder/adapter and Qwen decoder backward on asymmetric NCCL ranks."""
import os
import argparse
from datetime import timedelta
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
from latent_context.model import LCLM
from latent_context.encoder import Encoder
from latent_context.adapter import Adapter
from train.optimizer_utils import global_has_memory, clear_optimizer_group_grads
from utils.nan_checks import has_non_finite_loss_and_gradients


class Harness(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
        core=object.__new__(LCLM)
        torch.nn.Module.__init__(core)
        core.encoder=Encoder(compression_ratio=4,max_length=128,
            encoder_model=Qwen3Model(config),encoder_tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=1),
            train_encoder=True,encoder_window_size=32,pooling='mean',max_encode_batch_size=1,
            accelerator=SimpleNamespace(num_processes=dist.get_world_size()))
        core.adapter=Adapter(64,64)
        core.accelerator=None
        self.core=core
        self.decoder=Qwen3ForCausalLM(config)

    def forward(self,has_memory):
        embeddings,zero=self.core._process_batched_latent_embeddings(
            [[[3,4,5,6,7,8,9,10]*12]] if has_memory else [[]],ensure_participation=True)
        ids=torch.tensor([[2,3,4,5,6,7,8,9]],device=next(self.parameters()).device)
        inputs=self.decoder.get_input_embeddings()(ids)
        if has_memory:
            inputs=torch.cat([inputs[:,:2],embeddings[0][0][:2].unsqueeze(0),inputs[:,4:]],dim=1)
        loss=self.decoder(inputs_embeds=inputs,labels=ids).loss
        return loss+(zero if zero is not None else 0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fsdp',action='store_true')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    try:
        harness=Harness().cuda(rank)
        if args.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # Separate sharded modules exercise matching forward/backward
            # collectives across unequal encoder-window counts and empty ranks.
            harness.core.encoder.embed_model=FSDP(harness.core.encoder.embed_model,device_id=rank,use_orig_params=True)
            harness.core.adapter=FSDP(harness.core.adapter,device_id=rank,use_orig_params=True)
            model=FSDP(harness,device_id=rank,use_orig_params=True)
        else:model=DDP(harness,device_ids=[rank])
        opt=torch.optim.AdamW([
            {'params':harness.decoder.parameters()},
            {'params':harness.core.encoder.parameters()},
            {'params':harness.core.adapter.parameters()}],lr=1e-3)
        for pattern in ('all','rank0','none','rank1','none','all'):
            opt.zero_grad(set_to_none=True)
            has=pattern=='all' or pattern==f'rank{rank}'
            before=[p.detach().clone() for p in harness.core.parameters()]
            loss=model(has)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None for p in model.parameters() if p.numel())
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            any_memory=global_has_memory(has,device=torch.device('cuda',rank),distributed=True)
            if not any_memory:clear_optimizer_group_grads(opt,[1,2])
            opt.step()
            if not any_memory:
                assert all(torch.equal(a,b) for a,b in zip(before,harness.core.parameters()))
            dist.barrier()
            if rank==0:print(f'PASS NCCL {pattern}: loss={loss.item():.6f}',flush=True)
        def reduce_flag(value, reduction):
            assert reduction == 'sum'
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            return value
        accelerator=SimpleNamespace(device=torch.device('cuda',rank), reduce=reduce_flag)
        for bad in ('loss', 'gradient', 'neither'):
            opt.zero_grad(set_to_none=True)
            loss=model(rank == 0)
            loss.backward()
            checked_loss=loss.detach().clone()
            if rank==0 and bad=='loss':checked_loss.fill_(float('nan'))
            if rank==0 and bad=='gradient':
                next(p for p in model.parameters() if p.grad is not None and p.grad.numel()).grad.fill_(float('inf'))
            found=has_non_finite_loss_and_gradients(loss=checked_loss,model=model,accelerator=accelerator)
            assert found == (bad != 'neither')
            dist.barrier()
            if rank==0:print(f'PASS synchronized non-finite check: {bad}',flush=True)
    finally:
        dist.destroy_process_group()

if __name__=='__main__':main()
