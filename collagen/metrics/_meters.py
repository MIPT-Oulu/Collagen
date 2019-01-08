from ..core import Callback
import numpy as np


class RunningAverageMeter(Callback):
    def __init__(self):
        super(RunningAverageMeter, self).__init__()
        self.__value = 0
        self.__count = 0

    def on_epoch_begin(self, **kwargs):
        self.__value = 0
        self.__count = 0

    def on_batch_end(self, loss, **kwargs):
        self.__value += loss
        self.__count += 1

    def current(self):
        if self.__count == 0:
            return self.__value
        return self.__value / self.__count


class AccuracyMeter(Callback):
    def __init__(self):
        super(AccuracyMeter, self).__init__()
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_epoch_begin(self, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_batch_end(self, target, output, **kwargs):
        n = target.shape[0]
        output = output.argmax(dim=-1).view(n, -1)
        target = target.view(n, -1)

        acc = (output == target).float().sum()
        print("acc : {}".format(acc))
        return acc

    def update_metrics(self, target_on_cpu, pred_on_cpu):
        if pred_on_cpu.shape != target_on_cpu.shape:
            raise ValueError("Target shape ({}) and prediction ({}) shape mismatched.".format(target_on_cpu.shape, pred_on_cpu.shape))

        mask_matched = np.abs(target_on_cpu-pred_on_cpu) < self.__accuracy_threshold
        self.__correct_count += np.sum(mask_matched)
        self.__data_count += target_on_cpu.shape[0]

    def get_metrics(self):
        if self.__data_count > 0:
            acc = self.__correct_count/self.__data_count
        else:
            acc = -1.0
        # print("Accuracy = {}".format(acc))
        return acc