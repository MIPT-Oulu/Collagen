from abc import abstractmethod
import gc


class BaseMeter(object):
    def __init__(self, init_count=0, track_history=False):
        self.__count = init_count
        self.__track_history = track_history
        self.__history = list()

    def update(self, value):
        if self.__track_history is not None:
            self.__history.append(value)
        self.count += 1

    @abstractmethod
    def current(self):
        raise NotImplementedError

    def count(self):
        return self.__count

    def reset(self):
        self.__count = None
        del self.__history
        self.__history = None
        gc.collect()


class CumulativeMeter(BaseMeter):
    def __init__(self, init_val=None, init_count=None, track_history=False):
        super(CumulativeMeter, self).__init__(track_history)
        self.__value = init_val
        self.__count = init_count

    def update(self, value):
        super(CumulativeMeter, self).update(value)
        self.__value += value

    def current(self):
        return self.__value


class RunningAverageMeter(CumulativeMeter):
    def __init__(self, init_val=0, init_count=0):
        super(RunningAverageMeter, self).__init__(init_val, init_count, track_history=False)

    def current(self):
        if self.__count == 0:
            return self.__value
        return self.current() / self.__count

