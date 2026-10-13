"""Small optimizer helpers shared by the trainer and dependency-light tests."""

from collections.abc import Iterable

import torch


def global_has_memory(
    local_has_memory: bool,
    *,
    device: torch.device,
    distributed: bool,
) -> bool:
    """Return whether any rank used a real compressed memory region."""
    value = torch.tensor(int(local_has_memory), device=device, dtype=torch.int64)
    if distributed:
        import torch.distributed as dist

        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def clear_optimizer_group_grads(
    optimizer: torch.optim.Optimizer,
    group_indices: Iterable[int],
) -> None:
    """Set selected parameter grads to None so AdamW/state/decay do not advance."""
    for group_index in group_indices:
        for parameter in optimizer.param_groups[group_index]["params"]:
            parameter.grad = None

