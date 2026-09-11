"""Keep LCLM's gradient checks before DeepSpeed's optimizer update.

Accelerate's DeepSpeed backward wrapper also calls engine.step(). We instead use
the engine's separate backward/step API. ZeRO installs gradients on FP32 master
partitions inside step(), so the compression guard belongs on the underlying
optimizer's pre-step hook, after that installation. Do not clear model .grad and
assume it affects ZeRO's master gradients.
"""
from itertools import chain

import torch

from train.optimizer_utils import global_has_memory
from utils.nan_checks import has_non_finite_loss_and_gradients


COMPRESSION_GROUP = "lclm_compression_group"


def validate_zero_stage(stage):
    if stage not in (0, 1, 2):
        raise ValueError("LCLM training supports at most ZeRO-2; set zero_stage=2.")


class DeepSpeedStep:
    def __init__(self, engine, accelerator):
        self.engine = engine
        self.accelerator = accelerator
        self.optimizer = engine.optimizer
        self.stage = int(engine.zero_optimization_stage())
        validate_zero_stage(self.stage)
        self.invalid_window = False
        self.skip_compression = False
        self.last_step_applied = False
        if getattr(engine, "scale_wrt_gas", None) not in (None, False):
            raise NotImplementedError("LCLM owns loss scaling across DeepSpeed accumulation microbatches.")
        if self.stage in (1, 2):
            required = ("averaged_gradients", "single_partition_of_fp32_groups", "micro_step_id")
        elif engine.bfloat16_enabled():
            required = ("fp32_groups_gradients_flat",)
        elif engine.fp16_enabled():
            raise NotImplementedError("LCLM DeepSpeed FP16 without ZeRO is not supported; use ZeRO or BF16.")
        else:
            required = ()
        for name in required:
            if not hasattr(self.optimizer, name):
                raise RuntimeError(f"Unsupported DeepSpeed gradient layout: ZeRO-{self.stage} lacks {name}")
        if getattr(self.optimizer, "swap_optimizer", False):
            raise NotImplementedError("LCLM's pre-update gradient checks do not support NVMe optimizer swapping.")
        self.masked_gradients = []
        self.hooks = [
            engine.basic_optimizer.register_step_pre_hook(self._guard_compression),
            engine.basic_optimizer.register_step_post_hook(self._restore_gradients),
        ]

    def _guard_compression(self, optimizer, args, kwargs):
        if self.skip_compression:
            for group in optimizer.param_groups:
                if group.get(COMPRESSION_GROUP, False):
                    for parameter in group["params"]:
                        self.masked_gradients.append((parameter, parameter.grad))
                        parameter.grad = None

    def _restore_gradients(self, optimizer, args, kwargs):
        # ZeRO CPU offload reuses the master's gradient allocation on the next
        # backward. Suppress it only during the base optimizer's update.
        for parameter, gradient in self.masked_gradients:
            parameter.grad = gradient
        self.masked_gradients.clear()

    def begin_microbatch(self):
        # Must precede forward too, because ZeRO's hooks consult this boundary.
        self.engine.set_gradient_accumulation_boundary(self.accelerator.sync_gradients)

    def gradient_buffers(self):
        """Read local accumulation/partition gradients without gathering weights."""
        for parameter in self.engine.module.parameters():
            yield parameter.grad
            yield getattr(parameter, "grad_accum", None)
        if self.stage in (1, 2):
            if self.optimizer.cpu_offload:
                yield from self.optimizer.accumulated_grads_in_cpu.values()
                for parameter in self.optimizer.single_partition_of_fp32_groups:
                    yield parameter.grad
            else:
                yield from chain.from_iterable(values or () for values in self.optimizer.averaged_gradients.values())
        elif hasattr(self.optimizer, "fp32_groups_gradients_flat"):
            yield from self.optimizer.fp32_groups_gradients_flat

    @torch.no_grad()
    def _discard_window(self):
        # ZeRO.zero_grad() clears model gradients, but not all accumulation
        # buffers. A rejected window must not contaminate the next valid window.
        for gradient in self.gradient_buffers():
            if gradient is not None:
                gradient.zero_()
        self.optimizer.zero_grad()
        if self.engine.fp16_enabled() and self.stage in (1, 2):
            # We bypass the rejected optimizer step, but must still lower a
            # dynamic FP16 loss scale so subsequent finite windows can recover.
            self.optimizer._update_scale(True)
        if self.stage in (1, 2):
            self.optimizer.averaged_gradients.clear()
            from deepspeed.runtime.zero.stage_1_and_2 import INITIAL_MICRO_STEP_ID
            self.optimizer.micro_step_id = INITIAL_MICRO_STEP_ID
            if self.optimizer.cpu_offload:
                self.optimizer.reset_cpu_buffers()
            tracker = getattr(self.optimizer, "inf_or_nan_tracker", None)
            if tracker is not None:
                tracker.zero_()
        # Advance the engine's microbatch bookkeeping without an optimizer or
        # scheduler update. The next begin_microbatch restores the real boundary.
        self.engine.set_gradient_accumulation_boundary(False)
        self.engine.step()
        self.engine.skipped_steps += 1
        self.engine.losses = None

    def backward_and_step(self, loss, *, local_has_memory):
        self.last_step_applied = False
        # ZeRO-1's no_sync context disables DeepSpeed's automatic GAS scaling.
        # Scale each microbatch exactly once regardless of its reduction mode.
        self.engine.backward(loss.float() / self.engine.gradient_accumulation_steps(), scale_wrt_gas=False)
        invalid = has_non_finite_loss_and_gradients(
            loss=loss, model=self.engine, accelerator=self.accelerator,
            gradients=self.gradient_buffers())
        # Do not short-circuit the collective above when an earlier microbatch
        # failed. Every rank still participates on every microbatch.
        self.invalid_window = self.invalid_window or invalid
        if not self.accelerator.sync_gradients:
            return not self.invalid_window
        if self.invalid_window:
            self._discard_window()
            self.invalid_window = False
            return False
        has_memory = global_has_memory(local_has_memory, device=self.accelerator.device,
                                       distributed=self.accelerator.num_processes > 1)
        self.skip_compression = not has_memory
        try:
            self.engine.step()
        finally:
            self.skip_compression = False
        self.last_step_applied = not bool(getattr(self.optimizer, "overflow", False))
        return self.last_step_applied
