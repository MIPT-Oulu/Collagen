import numpy as np
from torch import optim
from typing import Tuple

from collagen.callbacks.lr_scheduling.utils import ramps
from collagen.core import Callback

__all__ = ["CosineAnnealingWarmRestartsWithWarmup", "LRScheduler", "SimpleLRScheduler", "SingleRampUpDownScheduler",
           "TemporalBasedScheduler", "MultiLinearByBatchScheduler", "MultiLinearByEpochScheduler"]


class LRScheduler(Callback):
    def __init__(self, name='lr_scheduler'):
        super().__init__(ctype='lrs', name=name)


class SimpleLRScheduler(LRScheduler):
    def __init__(self, metric_name, lr_scheduler):
        super().__init__(name="simple_lrs")
        self.__lr_scheduler: optim = lr_scheduler
        self.__metric_name: str = metric_name

    @property
    def metric_name(self):
        return self.__metric_name

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            if cb.ctype == "meter" and cb.desc == self.metric_name:
                self.__lr_scheduler.step(cb.current())


class CosineAnnealingWarmRestartsWithWarmup(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lrs, T_O, T_mult=1, eta_min=0, last_epoch=-1):
        super().__init__()
        if isinstance(warmup_lrs, float) and isinstance(warmup_epochs, int):
            self._warmup_lrs = (0, warmup_lrs)
            self._warmup_epochs = (0, warmup_epochs)
        elif len(warmup_epochs) == len(warmup_epochs):
            self._warmup_lrs = warmup_lrs
            self._warmup_epochs = warmup_epochs
        else:
            raise ValueError(f'`warmup_epochs` and `warmup_lrs` lengths must be matched, but got {len(warmup_epochs), len(warmup_lrs)}')

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', warmup_lrs[-1])

        self._last_epoch = warmup_epochs[-1]
        self._lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_O, T_mult, eta_min, self._last_epoch)

        self._optimizer = optimizer
        self._T_O = T_O
        self._T_mult = T_mult
        self._eta_min = eta_min


    def _check_valid(self, stage, *args):
        return stage == 'train'

    def on_batch_end(self, stage, n_batches, batch_i, epoch, *args, **kwargs):
        if self._check_valid(stage, n_batches, batch_i, epoch):
            if epoch <= self._warmup_epochs[-1]:
                lr = np.interp(epoch + batch_i / n_batches, self._warmup_epochs, self._warmup_lrs)
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self._lr_scheduler.step(epoch + batch_i / n_batches)


class TemporalBasedScheduler(LRScheduler):
    def __init__(self, lr_scheduler, name='single_ramupdown_lrs'):
        super().__init__(name=name)
        self.__lr_scheduler = lr_scheduler

    def on_batch_begin(self, *args, **kwargs):
        self.__lr_scheduler(*args, **kwargs)


class MultiLinearByEpochScheduler(LRScheduler):
    def __init__(self, optimizer, steps, lrs, scale=1.0, name="multi_linear_epoch_lrs"):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__steps = steps
        self.__lrs = lrs
        self.__scale = scale

    def on_epoch_begin(self, epoch, *args, **kwargs):
        lr = np.interp(epoch + 1, self.__steps, self.__lrs) * self.__scale
        print(f'LR: {lr}')
        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr


class MultiLinearByBatchScheduler(LRScheduler):
    def __init__(self, optimizer, n_batches, steps, lrs, scale=1.0, name="multi_linear_batch_lrs"):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__steps = steps
        self.__lrs = lrs
        self.__scale = scale
        self.__n_batches = n_batches
        self.__batch_i = 0

    def on_batch_begin(self, batch_i, *args, **kwargs):
        self.__batch_i += 1
        step = self.__batch_i / self.__n_batches
        lr = np.interp(step, self.__steps, self.__lrs) * self.__scale
        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr


class SingleRampUpDownScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr, rampup_epochs, lr, rampdown_epochs, name='single_rampupdown_lrs'):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__initial_lr = initial_lr
        self.__rampup_epochs = rampup_epochs
        self.__lr = lr
        self.__rampdown_epochs = rampdown_epochs

    def on_batch_begin(self, epoch, batch_i, n_batches, n_epochs, *args, **kwargs):
        if self.__rampdown_epochs <= n_epochs:
            raise ValueError(
                f'lr_rampdown_epochs {self.__rampdown_epochs} must larger than num of epochs {n_epochs}')
        epoch = epoch + batch_i / n_batches

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, self.__rampup_epochs) * (self.__lr - self.__initial_lr) + self.__initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.__rampdown_epochs:
            lr *= ramps.cosine_rampdown(epoch, self.__rampdown_epochs)

        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr


class CycleRampUpDownScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr, rampup_epochs, rampup_lr, start_cycle_epoch, rampdown_epochs,
                 cycle_rampdown_epochs, cycle_interval, constant_lr_epoch=10, constant_lr=None,
                 name='cycle_rampupdown_lrs'):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__initial_lr = initial_lr
        self.__rampup_epochs = rampup_epochs
        self.__rampup_lr = rampup_lr
        self.__rampdown_epochs = rampdown_epochs
        self.__constant_lr = constant_lr
        self.__constant_lr_epoch = constant_lr_epoch
        self.__start_cycle_epoch = start_cycle_epoch
        self.__cycle_rampdown_epochs = cycle_rampdown_epochs
        self.__cycle_interval = cycle_interval

    def on_batch_begin(self, epoch, n_epochs, batch_i, n_batches, *args, **kwargs):
        # if self.__rampdown_epochs <= n_epochs:
        #     raise ValueError(f'rampdown_epochs {self.__rampdown_epochs} must larger than num of epochs {n_epochs}')

        epoch = epoch + batch_i / n_batches
        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = np.interp(epoch, [0, self.__rampup_epochs], [self.__initial_lr, self.__rampup_lr])

        if self.__rampdown_epochs:
            if epoch < self.__start_cycle_epoch:
                # Cosine LR rampdown from https://arxiv.org/abs/1608.03983
                assert self.__rampdown_epochs >= self.__start_cycle_epoch
                lr *= ramps.cosine_rampdown(epoch, self.__rampdown_epochs)
            elif epoch >= self.__start_cycle_epoch:
                if self.__constant_lr:
                    constant_lr = ramps.cosine_rampdown(self.__constant_lr_epoch, self.__rampdown_epochs)
                    lr *= constant_lr
                else:
                    lr_rampdown_epochs = self.__rampdown_epochs if self.__cycle_rampdown_epochs == 0 else self.__cycle_rampdown_epochs
                    lr *= ramps.cosine_rampdown((lr_rampdown_epochs - (
                            self.__rampdown_epochs - self.__start_cycle_epoch) - self.__cycle_interval) + (
                                                        (epoch - self.__start_cycle_epoch) % self.__cycle_interval),
                                                lr_rampdown_epochs)

        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr
