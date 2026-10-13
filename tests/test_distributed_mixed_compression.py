import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from latent_context.model import LCLM
from train.optimizer_utils import global_has_memory


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.pad_token_id = 0

    def forward(self, segments):
        return [
            self.scale
            * torch.tensor([[float(sum(segment)), float(len(segment))]])
            for segment in segments
        ]


class TinyAdapter(nn.Module):
    adapter_type = "mlp"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        nn.init.eye_(self.fc1.weight)

    def forward(self, embeddings):
        return self.fc1(embeddings)


class MixedCompressionHarness(nn.Module):
    """Exercise LCLM's real/dummy path through an actual DDP reducer."""

    def __init__(self):
        super().__init__()
        core = object.__new__(LCLM)
        nn.Module.__init__(core)
        core.encoder = TinyEncoder()
        core.adapter = TinyAdapter()
        core.accelerator = None
        self.core = core

    def forward(self, has_memory):
        metadata = [[[1, 2, 3]]] if has_memory else [[]]
        embeddings, zero_dependency = self.core._process_batched_latent_embeddings(
            metadata,
            ensure_participation=True,
        )
        if has_memory:
            return embeddings[0][0].sum()
        return zero_dependency


def _distributed_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(MixedCompressionHarness())

        # Exercise both asymmetric orders. The empty rank must still trigger all
        # reducer hooks through the connected zero-valued dummy graph.
        for memory_rank in (1, 0):
            model.zero_grad(set_to_none=True)
            model(rank == memory_rank).backward()
            encoder_grad = model.module.core.encoder.scale.grad
            adapter_grad = model.module.core.adapter.fc1.weight.grad
            assert encoder_grad is not None and torch.count_nonzero(encoder_grad) > 0
            assert adapter_grad is not None and torch.count_nonzero(adapter_grad) > 0

        # Globally empty still gives zero, rather than missing, gradients.
        model.zero_grad(set_to_none=True)
        model(False).backward()
        encoder_grad = model.module.core.encoder.scale.grad
        adapter_grad = model.module.core.adapter.fc1.weight.grad
        assert encoder_grad is not None and torch.count_nonzero(encoder_grad) == 0
        assert adapter_grad is not None and torch.count_nonzero(adapter_grad) == 0

        assert global_has_memory(
            rank == 0, device=torch.device("cpu"), distributed=True
        )
        assert not global_has_memory(
            False, device=torch.device("cpu"), distributed=True
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_ddp_handles_asymmetric_empty_memory_in_both_orders():
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "gloo-init")
        mp.spawn(
            _distributed_worker,
            args=(2, init_file),
            nprocs=2,
            join=True,
        )

