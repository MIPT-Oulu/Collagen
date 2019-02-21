from torch import optim
from collagen.core import Callback


class LRScheduler(Callback):
    def __init__(self, metric_name, pt_lr_scheduler):
        super().__init__(type="lr_scheduler")
        self.__lr_scheduler: optim = pt_lr_scheduler
        self.__metric_name: str = metric_name

    @property
    def metric_name(self):
        return self.__metric_name

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            if cb.get_type() == "meter" and cb.get_name() == self.metric_name:
                self.__lr_scheduler.step(cb.current())
