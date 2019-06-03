from torch import optim
from collagen.core import Callback


class LRScheduler(Callback):
    def __init__(self, metric_name, lr_scheduler):
        super().__init__(ctype="lr_scheduler", name="lr_scheduler")
        self.__lr_scheduler: optim = lr_scheduler
        self.__metric_name: str = metric_name

    @property
    def metric_name(self):
        return self.__metric_name

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            if cb.ctype == "meter" and cb.desc == self.metric_name:
                self.__lr_scheduler.step(cb.current())
