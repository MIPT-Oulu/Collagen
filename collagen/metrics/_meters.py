from collagen.core import Callback
from sklearn.metrics import cohen_kappa_score
from collagen.data.utils import to_cpu


class Meter(Callback):
    def __init__(self, name: str = "unknown", prefix: str = ""):
        super().__init__(type="meter")
        self.__name = name
        self.__prefix = prefix

    def current(self):
        return None

    def __str__(self):
        name = self.get_name()
        value = self.current()
        return "{0}: {1:.3f}".format(name, value)

    def get_name(self):
        return self.__prefix + ("/" if self.__prefix else "") + self.__name


class RunningAverageMeter(Meter):
    def __init__(self, name: str = "loss", prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__value = 0
        self.__count = 0
        self.__avg_loss = None

    def on_epoch_begin(self, epoch, **kwargs):
        self.__value = 0
        self.__count = 0

    def on_minibatch_end(self, loss, **kwargs):
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
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
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

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)


class AccuracyThresholdMeter(Meter):
    def __init__(self, name: str = "binary_accuracy", threshold: float = 0.5, sigmoid: bool = False, prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__threshold: int = threshold
        self.__sigmoid: bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
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

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)


class SSAccuracyMeter(Meter):
    def __init__(self, name: str = "ssl_accuracy", sigmoid: bool = False, prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__sigmoid: bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        if len(target.shape) > 1 and target.shape[1] > 1:
            n = target.shape[0]
            target_cls = target[:,:-1].float()
            output_cls = output[:,:-1].float()
            target_valid = target[:, -1].float()
            if device is None:
                device = output.device
                target_on_device = target_cls.to(device)
                target_valid_on_device = target_valid.to(device)
                output_on_device = output_cls
            else:
                target_on_device = target_cls.to(device)
                target_valid_on_device = target_valid.to(device)
                output_on_device = output_cls.to(device)

            if self.__sigmoid:
                output_on_device = output_on_device.sigmoid()

            discrete_output_on_device = output_on_device.argmax(dim=-1).view(n)
            discrete_target_on_device = target_on_device.argmax(dim=-1).view(n)

            cls = (discrete_output_on_device.byte() == discrete_target_on_device.byte()).float()
            fp = (target_valid_on_device*cls).sum()
            total = target_valid_on_device.sum().float()
            self.__correct_count += fp
            self.__data_count += total

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)


class SSValidityMeter(Meter):
    def __init__(self, name: str = "ssl_validity", threshold: float = 0.5, sigmoid: bool = False, prefix=""):
        super().__init__(name=name, prefix=prefix)
        self.__sigmoid: bool = sigmoid
        self.__threshold = threshold
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        if len(target.shape) > 1 and target.shape[1] > 1:
            target_valid = target[:,-1]
        elif len(target.shape) == 1:
            target_valid = target
        else:
            raise ValueError("Not support target shape like {}".format(target.shape))

        if len(output.shape) > 1 and output.shape[1] > 1:
            output_valid = output[:,-1]
        elif len(output.shape) == 1:
            output_valid = output
        else:
            raise ValueError("Not support output shape like {}".format(output.shape))

        n = target.shape[0]

        if device is None:
            device = output.device
            target_on_device = target_valid.to(device)
            output_on_device = output_valid
        else:
            target_on_device = target_valid.to(device)
            output_on_device = output_valid.to(device)

        if self.__sigmoid:
            output_on_device = output_on_device.sigmoid()

        valid = ((output_on_device > self.__threshold) == target_on_device.byte()).float()
        fp = valid.sum()
        self.__correct_count += fp
        self.__data_count += n
        acc = self.__correct_count/self.__data_count
        acc1 = acc

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
        else:
            acc = 0.0
        return to_cpu(acc, use_numpy=True)


class KappaMeter(Meter):
    def __init__(self, name: str = "kappa", prefix="", weight_type="quadratic"):
        super().__init__(name=name, prefix=prefix)
        self.__predicts = []
        self.__corrects = []
        self.__kappa = None
        self.__weight_type = weight_type

    def on_epoch_begin(self, epoch, **kwargs):
        self.__predicts = []
        self.__corrects = []

    def on_minibatch_end(self, target, output, **kwargs):
        if len(target.shape) == 2:
            target_cpu = to_cpu(target.argmax(dim=1), use_numpy=True)
        elif len(target.shape) == 1:
            target_cpu = to_cpu(target, use_numpy=True)
        else:
            raise ValueError("Only support dims 1 or 2, but got {}".format(len(target.shape)))

        if len(output.shape) == 2:
            output_cpu = to_cpu(output.argmax(dim=1), use_numpy=True)
        elif len(output.shape) == 1:
            output_cpu = to_cpu(output, use_numpy=True)
        else:
            raise ValueError("Only support dims 1 or 2, but got {}".format(len(output.shape)))

        self.__predicts += output_cpu.tolist()
        self.__corrects += target_cpu.tolist()

    def on_epoch_end(self, *args, **kwargs):
        self.__kappa = self.current()

    def current(self):
        if len(self.__corrects) != len(self.__predicts):
            raise ValueError("Predicts and corrects must match, but got {} vs {}".format(len(self.__corrects), len(self.__predicts)))
        return cohen_kappa_score(self.__corrects, self.__predicts, weights=self.__weight_type)