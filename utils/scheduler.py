# put this somewhere importable, e.g. utils/schedulers.py
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithMinLRScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, cosine_fraction=1.0, last_epoch=-1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_lr = min_lr  # single min_lr value for all parameter groups
        self.cosine_fraction = cosine_fraction  # fraction of training where cosine decay happens
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.num_warmup_steps:
            warm = step / max(1, self.num_warmup_steps)
            return [base_lr * warm for base_lr in self.base_lrs]

        # Calculate the number of steps for cosine decay
        cosine_steps = int((self.num_training_steps - self.num_warmup_steps) * self.cosine_fraction)
        cosine_end_step = self.num_warmup_steps + cosine_steps
        
        if step <= cosine_end_step:
            # Cosine decay phase
            prog = (step - self.num_warmup_steps) / max(1, cosine_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
        else:
            # Constant min_lr phase
            return [self.min_lr for _ in self.base_lrs]
