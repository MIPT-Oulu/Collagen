from ..core import Callback
from collagen.data.utils import to_cpu


class Meter(Callback):
    def __init__(self, name: str = "unknown", prefix: str = ""):
        super().__init__(type="meter")
        self.__name = name
        self.__prefix = prefix
        self.__metric = -1.0

    def current(self):
        return None

    def __str__(self):
        prefix = self.__prefix + ("_" if self.__prefix else "")
        return "{0}{1}: {2:.3f}".format(prefix, self.__name, self.current())


class RunningAverageMeter(Meter):
    def __init__(self, name: str = "loss", prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__name = name
        self.__value = 0
        self.__count = 0
        self.__metric = None

    def on_epoch_begin(self, epoch, **kwargs):
        if epoch == 0:
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


class AccuracyMeter(Meter):
    def __init__(self, name: str = "categorical_accuracy", prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__name = name
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        if epoch == 0:
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

    def on_epoch_end(self, epoch, num_epochs, *args, **kwargs):
        if epoch >= num_epochs - 1:
            self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count/self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)


class AccuracyThresholdMeter(Meter):
    def __init__(self, name: str = "binary_accuracy", threshold:float = 0.5, sigmoid:bool = False, prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__name: str = name
        self.__threshold:int = threshold
        self.__sigmoid:bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        if epoch == 0:
            self.__data_count = 0.0
            self.__correct_count = 0.0

    def on_batch_end(self, target, output, device=None, **kwargs):
        n = target.shape[0]

        if device is None:
            device = output.device
            target_on_device = target.to(device)
            output_on_device = output
        else:
            target_on_device = target.to(device)
            output_on_device = output.to(device)

        if self.__sigmoid:
            output_on_device = output_on_device.sigmoid()

        self.__correct_count += ((output_on_device > self.__threshold) == target_on_device.byte()).float().sum()
        self.__data_count += n

    def on_epoch_end(self, epoch, num_epochs, *args, **kwargs):
        if epoch >= num_epochs - 1:
            self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count/self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)
