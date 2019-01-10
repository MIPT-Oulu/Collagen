from ..core import Callback


class RunningAverageMeter(Callback):
    def __init__(self):
        super(RunningAverageMeter, self).__init__()
        self.__value = 0
        self.__count = 0
        self.__avg_loss = None

    def on_epoch_begin(self, **kwargs):
        self.__value = 0
        self.__count = 0

    def on_batch_end(self, loss, **kwargs):
        self.__value += loss
        self.__count += 1

    def on_epoch_end(self, *args, **kwargs):
        self.__avg_loss = self.current()

    def current(self):
        if self.__count == 0:
            return self.__value
        return self.__value / self.__count


class AccuracyMeter(Callback):
    def __init__(self):
        super(AccuracyMeter, self).__init__()
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_batch_end(self, target, output, device=None, **kwargs):
        n = target.shape[0]
        output = output.argmax(dim=-1).view(n, -1)
        target = target.view(n, -1)

        if device is None:
            device = output.device
            target_on_device = target.to(device)
            output_on_device = output
        else:
            target_on_device = target.to(device)
            output_on_device = output.to(device)
        self.__correct_count += (output_on_device == target_on_device).float().sum()
        self.__data_count += n

    def on_epoch_end(self, *args, **kwargs):
        self.__accuracy = self.current()
        # print("Accuracy: {}".format(acc))

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count/self.__data_count
        else:
            acc = -1.0
        return acc
