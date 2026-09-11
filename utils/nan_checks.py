#!/usr/bin/env python3
"""
NaN/Inf loss and gradient checks utilities
"""

from collections import defaultdict
import torch


def has_non_finite_loss_and_gradients(*, loss: torch.Tensor, model, accelerator, gradients=None) -> bool:
    """Return True if any rank sees non-finite loss or gradients.

    - Checks local loss and gradients for non-finite values.
    - Uses distributed reduction via the provided accelerator to synchronize the decision.
    """
    # Keep the scan on device, without a host synchronization per parameter.
    # Infinity norms cannot overflow from adding otherwise finite gradients.
    groups = defaultdict(list)
    if gradients is None:
        gradients = (param.grad for param in model.parameters())
    for gradient in gradients:
        if gradient is not None and gradient.numel():
            grad = gradient.detach()
            if grad.is_sparse:
                grad = grad.coalesce().values()
            groups[(grad.device, grad.dtype)].append(grad)
    flags = [~torch.isfinite(loss.detach()).all().to(accelerator.device)]
    for gradients in groups.values():
        norms = torch._foreach_norm(gradients, float('inf'))
        flags.append(~torch.isfinite(torch.stack(norms)).all().to(accelerator.device))
    local_flag = torch.stack(flags).any().to(dtype=torch.int32)
    # Every rank must call this, even when its own loss is finite. Never hide a
    # failed collective by falling back to a rank-local optimizer decision.
    return accelerator.reduce(local_flag, reduction="sum").item() > 0
