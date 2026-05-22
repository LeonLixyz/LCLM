#!/usr/bin/env python3
"""
NaN/Inf loss and gradient checks utilities
"""

from typing import Tuple
import torch


def has_non_finite_loss_and_gradients(*, loss: torch.Tensor, model, accelerator) -> bool:
    """Return True if any rank sees non-finite loss or gradients.

    - Checks local loss and gradients for non-finite values.
    - Uses distributed reduction via the provided accelerator to synchronize the decision.
    """
    # Local checks
    non_finite_loss_local = not torch.isfinite(loss).item()

    non_finite_grad_local = False
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(
                f"[rank {accelerator.process_index}] Non-finite gradient in {name} shape={tuple(param.grad.shape)}",
                flush=True,
            )
            non_finite_grad_local = True
            break

    if non_finite_loss_local:
        print(
            f"[rank {accelerator.process_index}] Non-finite loss detected: {loss.item()}",
            flush=True,
        )

    # Sync decision across ranks
    try:
        loss_flag_tensor = torch.tensor(
            1 if non_finite_loss_local else 0,
            device=accelerator.device,
            dtype=torch.int32,
        )
        grad_flag_tensor = torch.tensor(
            1 if non_finite_grad_local else 0,
            device=accelerator.device,
            dtype=torch.int32,
        )

        loss_any = accelerator.reduce(loss_flag_tensor, reduction="max").item() > 0
        grad_any = accelerator.reduce(grad_flag_tensor, reduction="max").item() > 0
        return loss_any or grad_any
    except Exception:
        return non_finite_grad_local or non_finite_loss_local


