from ..core import Callback


class RunningAverageMeter(Callback):
    def __init__(self):
        super(RunningAverageMeter, self).__init__()
        self.__value = 0
        self.__count = 0

    def on_epoch_begin(self):
        self.__value = 0
        self.__count = 0

    def on_batch_end(self, value):
        self.__value += value
        self.__count += 1

    def current(self):
        if self.__count == 0:
            return self.__value
        return self.__value / self.__count

    def on_batch_begin(self):
        pass

    def on_epoch_end(self):
        pass

