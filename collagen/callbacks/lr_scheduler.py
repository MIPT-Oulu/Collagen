from torch import optim
from collagen.core import Callback
from collagen.lrscheduler.utils import ramps


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


class TemporalBasedScheduler(LRScheduler):
    def __init__(self, lr_scheduler, name='single_ramupdown_lrs'):
        super().__init__(name=name)
        self.__lr_scheduler = lr_scheduler

    def on_batch_begin(self, *args, **kwargs):
        self.__lr_scheduler(*args, **kwargs)


class SingleRampUpDownScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr, lr_rampup, lr, lr_rampdown_epochs, name='single_rampupdown_lrs'):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__initial_lr = initial_lr
        self.__lr_rampup = lr_rampup
        self.__lr = lr
        self.__lr_rampdown_epochs = lr_rampdown_epochs

    def on_batch_begin(self, epoch, batch_i, n_batches, n_epochs, *args, **kwargs):
        if self.__lr_rampdown_epochs <= n_epochs:
            raise ValueError(f'lr_rampdown_epochs {self.__lr_rampdown_epochs} must larger than num of epochs {n_epochs}')
        epoch = epoch + batch_i / n_batches

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, self.__lr_rampup) * (self.__lr - self.__initial_lr) + self.__initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.__lr_rampdown_epochs:
            lr *= ramps.cosine_rampdown(epoch, self.__lr_rampdown_epochs)

        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr


class CycleRampUpDownScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr, lr_rampup, lr, lr_rampdown_epochs, start_cycle_epoch,
                 cycle_rampdown_epochs, cycle_interval, constant_lr_epoch = 10, constant_lr = None, name='cycle_rampupdown_lrs'):
        super().__init__(name=name)
        self.__optim = optimizer
        self.__initial_lr = initial_lr
        self.__lr_rampup = lr_rampup
        self.__lr = lr
        self.__lr_rampdown_epochs = lr_rampdown_epochs
        self.__constant_lr = constant_lr
        self.__constant_lr_epoch = constant_lr_epoch
        self.__start_cycle_epoch = start_cycle_epoch
        self.__cycle_rampdown_epochs = cycle_rampdown_epochs
        self.__cycle_interval = cycle_interval

    def on_batch_begin(self, epoch, n_epochs, batch_i, n_batches, *args, **kwargs):
        if self.__lr_rampdown_epochs <= n_epochs:
            raise ValueError(f'lr_rampdown_epochs {self.__lr_rampdown_epochs} must larger than num of epochs {n_epochs}')
        
        epoch = epoch + batch_i / n_batches
        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, self.__lr_rampup) * (self.__lr - self.__initial_lr) + self.__initial_lr

        if self.__lr_rampdown_epochs:
            if epoch < self.__start_cycle_epoch:
                # Cosine LR rampdown from https://arxiv.org/abs/1608.03983
                assert self.__lr_rampdown_epochs >= self.__start_cycle_epoch
                lr *= ramps.cosine_rampdown(epoch, self.__lr_rampdown_epochs)
            elif epoch >= self.__start_cycle_epoch:
                if self.__constant_lr:
                    constant_lr = ramps.cosine_rampdown(self.__constant_lr_epoch, self.__lr_rampdown_epochs)
                    lr *= constant_lr
                else:
                    lr_rampdown_epochs = self.__lr_rampdown_epochs if self.__cycle_rampdown_epochs == 0 else self.__cycle_rampdown_epochs
                    lr *= ramps.cosine_rampdown(
                        (lr_rampdown_epochs - (self.__lr_rampdown_epochs - self.__start_cycle_epoch) - self.__cycle_interval) +
                        ((epoch - self.__start_cycle_epoch) % self.__cycle_interval), lr_rampdown_epochs)

        for param_group in self.__optim.param_groups:
            param_group['lr'] = lr
