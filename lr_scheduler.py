"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

from bisect import bisect_right

import torch.optim.lr_scheduler


class LinearWarmupMultiStepDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, warmup_rate, decay_rate,
                 num_epochs, decay_epochs, iters_per_epoch, override_lr=0.,
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.warmup_rate = warmup_rate
        self.decay_rate = decay_rate
        self.decay_epochs = [decay_epoch * iters_per_epoch for decay_epoch in decay_epochs]
        self.num_epochs = num_epochs * iters_per_epoch
        self.override_lr = override_lr
        super(LinearWarmupMultiStepDecayLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * (self.warmup_rate + (1. - self.warmup_rate) * alpha) \
                    for base_lr in self.base_lrs]
        else:
            if self.override_lr > 0.:
                return [self.override_lr for _ in self.base_lrs]
            e = bisect_right(self.decay_epochs, self.last_epoch)
            return [base_lr * (self.decay_rate ** e) for base_lr in self.base_lrs]