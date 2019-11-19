import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score

from collagen.core import Callback
from collagen.core.utils import to_cpu


class Meter(Callback):
    def __init__(self, name: str = "unknown", prefix: str = "", desc_name=None):
        super().__init__(ctype="meter")
        self.__name = name
        self.__prefix = prefix
        self.__desc_name = desc_name

    def current(self):
        return None

    def __str__(self):
        name = self.desc
        value = self.current()
        return "{0}: {1:.3f}".format(name, value)

    def _query_loss(self, loss):
        if isinstance(loss, float):
            loss_value = loss
        elif isinstance(loss, tuple):
            if len(loss) > 1 and self.name in loss[1]:
                loss_value = loss[self.name]
            else:
                loss_value = loss[0]
        else:
            loss_value = None
        return loss_value

    @staticmethod
    def default_parse_class(self, x):
        if len(x.shape) == 2:
            output_cpu = to_cpu(x.argmax(dim=1), use_numpy=True)
        elif len(x.shape) == 1:
            output_cpu = to_cpu(x, use_numpy=True)
        else:
            raise ValueError("Only support dims 1 or 2, but got {}".format(len(x.shape)))
        return output_cpu

    @staticmethod
    def default_cond(target, output):
        return True

    @staticmethod
    def default_parse_target(target):
        return target

    @staticmethod
    def default_parse_output(output):
        return output

    @property
    def name(self):
        return self.__name

    @property
    def desc(self):
        return self.__prefix + (
            "/" if self.__prefix else "") + self.name if self.__desc_name is None else self.__desc_name

    def __str__(self):
        return self.desc


class RunningAverageMeter(Meter):
    def __init__(self, name: str = "loss", prefix="", desc_name=None):
        super().__init__(name=name, prefix=prefix, desc_name=desc_name)
        self.__value = 0
        self.__count = 0
        self.__avg_loss = None

    def on_epoch_begin(self, epoch, **kwargs):
        self.__value = 0
        self.__count = 0

    def on_minibatch_end(self, loss, session, **kwargs):
        if hasattr(session.loss, 'get_loss_by_name'):
            loss_value = session.loss.get_loss_by_name(self.name)
        else:
            loss_value = loss
        if loss_value is not None:
            self.__value += loss_value
            self.__count += 1

    def on_epoch_end(self, *args, **kwargs):
        self.__avg_loss = self.current()

    def current(self):
        if self.__count == 0:
            return None
        return self.__value / self.__count


class AccuracyMeter(Meter):
    def __init__(self, name: str = "categorical_accuracy", prefix="", parse_target=None, parse_output=None, cond=None):
        super().__init__(name=name, prefix=prefix)
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output
        self.__cond = Meter.default_cond if cond is None else cond

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        if self.__cond(target, output):
            target = self.__parse_target(target)
            output = self.__parse_output(output)
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
            self.__correct_count += (output_on_device.to(torch.int64) == target_on_device.to(torch.int64)).float().sum()
            self.__data_count += n

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
            return to_cpu(acc, use_numpy=True)
        else:
            return None


class BalancedAccuracyMeter(Meter):
    def __init__(self, name: str = "balanced_categorical_accuracy", prefix="", parse_target=None, parse_output=None,
                 cond=None):
        super().__init__(name=name, prefix=prefix)
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None
        self.__corrects = []
        self.__preds = []
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output
        self.__cond = Meter.default_cond if cond is None else cond

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__preds = []
        self.__corrects = []

    def _calc_metric(self, target, output, device=None, **kwargs):
        target = to_cpu(target, use_numpy=True)
        output = to_cpu(output, use_numpy=True)
        n = target.shape[0]
        if len(target.shape) > 1 and target.shape[1] > 1:
            target_cls_list = np.argmax(target, axis=-1).tolist()
        else:
            target_cls_list = target.tolist()

        output_cls_list = np.argmax(output, axis=-1).tolist()

        self.__corrects += target_cls_list
        self.__preds += output_cls_list

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        if self.__cond(target, output):
            target = self.__parse_target(target)
            output = self.__parse_output(output)
            self._calc_metric(target, output, device, **kwargs)

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if len(self.__corrects) > 0:
            return balanced_accuracy_score(y_true=self.__corrects, y_pred=self.__preds)
        else:
            return None


class AccuracyThresholdMeter(Meter):
    def __init__(self, name: str = "binary_accuracy", threshold: float = 0.5,
                 sigmoid: bool = False, prefix="",
                 parse_target=None, parse_output=None, batch_wise=False):
        super().__init__(name=name, prefix=prefix)
        self.__batch_wise = batch_wise
        self.__threshold: float = threshold
        self.__sigmoid: bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None

        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_begin(self, *args, **kwargs):
        if self.__batch_wise:
            self.__data_count = 0.0
            self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        n = target.shape[0]

        target = self.__parse_target(target)
        output = self.__parse_output(output)

        if device is None:
            device = output.device
            target_on_device = target.to(device)
            output_on_device = output
        else:
            target_on_device = target.to(device)
            output_on_device = output.to(device)

        if self.__sigmoid:
            output_on_device = output_on_device.sigmoid()

        self.__correct_count += ((output_on_device > self.__threshold) == target_on_device.bool()).float().sum()
        self.__data_count += n

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
            return to_cpu(acc, use_numpy=True)
        else:
            return None


class SSAccuracyMeter(Meter):
    def __init__(self, name: str = "ssl_accuracy", sigmoid: bool = False, prefix="",
                 cond=None, parse_target=None, parse_output=None):
        super().__init__(name=name, prefix=prefix)
        self.__sigmoid: bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None
        self.__cond = Meter.default_cond if cond is None else cond
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def _calc_metric(self, target, output, device=None, **kwargs):
        if self.__cond(target, output):
            target_cls, target_valid = self.__parse_target(target)
            output_cls, _ = self.__parse_output(output)

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

            cls = (discrete_output_on_device.long() == discrete_target_on_device.long()).float()
            fp = (target_valid_on_device * cls).sum()
            total = target_valid_on_device.sum().float()
            self.__correct_count += fp
            self.__data_count += total

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        self._calc_metric(target, output, device, **kwargs)

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
            to_cpu(acc, use_numpy=True)
        else:
            return None


class SSBalancedAccuracyMeter(Meter):
    def __init__(self, name: str = "ssl_balanced_accuracy", sigmoid: bool = False, prefix="",
                 cond=None, parse_target=None, parse_output=None):
        super().__init__(name=name, prefix=prefix)
        self.__sigmoid: bool = sigmoid
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None
        self.__corrects = []
        self.__preds = []
        self.__cond = Meter.default_cond if cond is None else cond
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__preds = []
        self.__corrects = []

    def _calc_metric(self, target, output, device=None, **kwargs):
        if self.__cond(target, output):
            target_cls, target_valid = self.__parse_target(target)
            output_cls, _ = self.__parse_output(output)

            target_cls = to_cpu(target_cls, use_numpy=True)
            target_valid = to_cpu(target_valid, use_numpy=True)
            output_cls = to_cpu(output_cls, use_numpy=True)

            mask = target_valid == 1

            target_cls = target_cls[mask, :]
            output_cls = output_cls[mask, :]

            target_cls_list = np.argmax(target_cls, axis=-1).tolist()
            output_cls_list = np.argmax(output_cls, axis=-1).tolist()

            self.__corrects += target_cls_list
            self.__preds += output_cls_list

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        self._calc_metric(target, output, device, **kwargs)

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if len(self.__corrects) > 0:
            return balanced_accuracy_score(y_true=self.__corrects, y_pred=self.__preds)
        else:
            return None


class SSValidityMeter(Meter):
    def __init__(self, name: str = "ssl_validity", threshold: float = 0.5, sigmoid: bool = False, prefix="",
                 cond=None, parse_target=None, parse_output=None):
        super().__init__(name=name, prefix=prefix)
        self.__sigmoid: bool = sigmoid
        self.__threshold = threshold
        self.__data_count = 0.0
        self.__correct_count = 0.0
        self.__accuracy = None
        self.__cond = Meter.default_cond
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__data_count = 0.0
        self.__correct_count = 0.0

    def on_minibatch_end(self, target, output, device=None, **kwargs):
        if self.__cond(target, output):
            target_valid = self.__parse_target(target)
            output_valid = self.__parse_output(output)

            n = target_valid.shape[0]

            if device is None:
                device = output.device
                target_on_device = target_valid.to(device)
                output_on_device = output_valid
            else:
                target_on_device = target_valid.to(device)
                output_on_device = output_valid.to(device)

            if self.__sigmoid:
                output_on_device = output_on_device.sigmoid()

            valid = ((output_on_device > self.__threshold) == target_on_device.bool()).float()
            fp = valid.sum()
            self.__correct_count += fp
            self.__data_count += n

    def on_epoch_end(self, epoch, n_epochs, *args, **kwargs):
        self.__accuracy = self.current()

    def current(self):
        if self.__data_count > 0:
            acc = self.__correct_count / self.__data_count
            return to_cpu(acc, use_numpy=True)
        else:
            return None


class KappaMeter(Meter):
    def __init__(self, name: str = "kappa", prefix="", weight_type="quadratic",
                 parse_target=None, parse_output=None, cond=None):
        super().__init__(name=name, prefix=prefix)
        self.__predicts = []
        self.__corrects = []
        self.__kappa = None
        self.__weight_type = weight_type

        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output
        self.__cond = Meter.default_cond if cond is None else cond

    def on_epoch_begin(self, epoch, **kwargs):
        self.__predicts = []
        self.__corrects = []

    def on_minibatch_end(self, target, output, **kwargs):
        if self.__cond(target, output):
            target_cpu = self.__parse_target(target)
            output_cpu = self.__parse_output(output)

            if target_cpu is not None and output_cpu is not None and target_cpu.shape == output_cpu.shape:
                self.__predicts += output_cpu.tolist()
                self.__corrects += target_cpu.tolist()

    def on_epoch_end(self, *args, **kwargs):
        self.__kappa = self.current()

    def current(self):
        if len(self.__corrects) != len(self.__predicts):
            raise ValueError("Predicts and corrects must match, but got {} vs {}".format(len(self.__corrects),
                                                                                         len(self.__predicts)))
        elif len(self.__corrects) == 0:
            return None
        else:
            return cohen_kappa_score(self.__corrects, self.__predicts, weights=self.__weight_type)


class ConfusionMeter(Meter):
    def __init__(self, n_classes, name='confusion_matrix', prefix="",
                 parse_target=None, parse_output=None, cond=None, class_dim=-1):

        super(ConfusionMeter, self).__init__(name=name, prefix=prefix)
        self._n_classes = n_classes
        self._confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.uint64)

        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output
        self.__cond = Meter.default_cond if cond is None else cond
        self._blocked = False
        self.class_dim = class_dim

    def switch_blocker(self):
        self._blocked = not self._blocked

    def is_blocked(self):
        return self._blocked

    def reset(self):
        self._confusion_matrix = np.zeros((self._n_classes, self._n_classes), dtype=np.uint64)

    def on_epoch_begin(self, *args, **kwargs):
        self.reset()

    def _compute_confusion_matrix(self, targets, predictions):
        """
        https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
        """

        replace_indices = np.vstack((
            targets.flatten(),
            predictions.flatten())
        ).T

        confusion_matrix, _ = np.histogramdd(
            replace_indices,
            bins=(self._n_classes, self._n_classes),
            range=[(0, self._n_classes), (0, self._n_classes)]
        )
        self._confusion_matrix += confusion_matrix.astype(np.uint64)

    def on_minibatch_end(self, target, output, **kwargs):
        target = self.__parse_target(target)
        output = self.__parse_output(output)

        if self.__cond(target, output):
            target_cpu = to_cpu(target, use_numpy=True)
            output_cpu = to_cpu(output, use_numpy=True).argmax(self.class_dim)

            if target_cpu is not None and output_cpu is not None and target_cpu.shape == output_cpu.shape:
                self._compute_confusion_matrix(target_cpu, output_cpu)

    def current(self):
        return self._confusion_matrix


class JaccardDiceMeter(Meter):
    def __init__(self, n_classes=2, prefix="", name='jaccard', parse_output=None, class_names=None,
                 parse_target=None, cond=None, confusion_matrix=None):
        super(JaccardDiceMeter, self).__init__(name=name, prefix=prefix, desc_name=None)
        assert name in ['jaccard', 'dice']

        if confusion_matrix is not None:
            # For efficiency, we can share the confusion meter among multiple meters
            self.confusion_matrix = confusion_matrix
            assert confusion_matrix.current().shape[0] == n_classes
        else:
            self.confusion_matrix = ConfusionMeter(n_classes=n_classes,
                                                   prefix="", parse_target=parse_target,
                                                   parse_output=parse_output, cond=cond, class_dim=1)

        self.updating_cm = True
        if class_names is None:
            self.class_names = list(range(n_classes))
        else:
            assert len(class_names) == n_classes
            self.class_names = class_names

    def on_minibatch_start(self,**kwargs):
        # Mechanism of blocking will allow to share 1 confusion matrix among several meters
        # For example, IoU and Dice. with different thresholds.
        if not self.confusion_matrix.is_blocked():
            self.updating_cm = True
            self.confusion_matrix.switch_blocker()
        else:
            self.updating_cm = False

    def on_epoch_begin(self, *args, **kwargs):
        self.confusion_matrix.on_epoch_begin(*args, *kwargs)

    def on_minibatch_end(self, target, output, **kwargs):

        self.confusion_matrix.on_minibatch_end(target, output)

    def current(self):
        if self.name == 'jaccrad':
            coeffs = self.compute_jaccard(self.confusion_matrix.current())
        else:
            coeffs = self.compute_dice(self.confusion_matrix.current())

        return {cls:res for cls, res in zip(self.class_names, coeffs)}

    @staticmethod
    def compute_jaccard(confusion_matrix):
        """
        https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
        """
        confusion_matrix = confusion_matrix.astype(float)
        ious = []
        for index in range(confusion_matrix.shape[0]):
            true_positives = confusion_matrix[index, index]
            false_positives = confusion_matrix[:, index].sum() - true_positives
            false_negatives = confusion_matrix[index, :].sum() - true_positives
            denom = true_positives + false_positives + false_negatives
            if denom == 0:
                iou = 0
            else:
                iou = float(true_positives) / denom
            ious.append(iou)
        return ious

    @staticmethod
    def compute_dice(confusion_matrix):
        """
        https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py
        """
        confusion_matrix = confusion_matrix.astype(float)
        dices = []
        for index in range(confusion_matrix.shape[0]):
            true_positives = confusion_matrix[index, index]
            false_positives = confusion_matrix[:, index].sum() - true_positives
            false_negatives = confusion_matrix[index, :].sum() - true_positives
            denom = 2 * true_positives + false_positives + false_negatives
            if denom == 0:
                dice = 0
            else:
                dice = 2 * float(true_positives) / denom
            dices.append(dice)
        return dices


class ItemWiseBinaryJaccardDiceMeter(Meter):
    """
    Implements device-invariant image-Wise Jaccard and Dice Meter for binary segmentation problems.
    If both target and out are on gpu, then the computations will happen there.

    Parameters
    ----------
    prefix: str
        Prefix to be displayed in the progressbar and tensorboard
    name: str
        Name of the metric. Can only be `jaccard` or `dice`
    parse_output: Callable
        Function to parse the output
    parse_target: Callable
        Function to parse the target tensor
    cond: Callable
        Condition under which the metric will be updated
    """
    def __init__(self, prefix="", name="jaccard",
                 parse_output=None, parse_target=None, cond=None, eps=1e-8):
        super(ItemWiseBinaryJaccardDiceMeter, self).__init__(name=name, prefix=prefix, desc_name=None)
        assert name in ['jaccard', 'dice']

        self.eps = eps
        self.__parse_target = Meter.default_parse_target if parse_target is None else parse_target
        self.__parse_output = Meter.default_parse_output if parse_output is None else parse_output
        self.__cond = Meter.default_cond if cond is None else cond
        self.__value = None
        self.__batch_count = None


    @staticmethod
    def compute_dice(target, output, eps=1e-8):
        if target.sum() == 0 and output.sum() == 0:
            return torch.tensor(1)
        num = output.size(0)
        m1 = output.view(num, -1).float()
        m2 = target.view(num, -1).float()

        a = (m1 * m2).sum(1).add(eps)
        b = (m1.sum(1) + m2.sum(1)).add(eps)

        result = a.mul(2).div(b)

        return result

    @staticmethod
    def compute_jaccard(self, target, output, eps=1e-8):
        d = self.compute_dice(target, output, eps)
        return d / (2 - d)

    def on_epoch_begin(self, *args, **kwargs):
        self.__value = None
        self.__batch_count = 0

    def on_minibatch_end(self, target, output, **kwargs):
        target = self.__parse_target(target)
        output = self.__parse_output(output)

        if self.__cond(target, output):
            if target is not None and output is not None and target.shape == output.shape:
                with torch.no_grad():
                    if self.name == 'dice':
                        if self.__value is not None:
                            self.__value += self.compute_dice(target, output, self.eps)
                        else:
                            self.__value = self.compute_dice(target, output, self.eps)
                    else:
                        if self.__value is not None:
                            self.__value += self.compute_jaccard(target, output, self.eps)
                        else:
                            self.__value = self.compute_jaccard(target, output, self.eps)
                    self.__batch_count += 1

    def current(self):
        if self.__batch_count == 0 or self.__value is None:
            return None
        return self.__value / self.__batch_count

class MultilabelDiceMeter(Meter):
    """
    Implements device-invariant image-Wise Jaccard and Dice Meter for binary multilabel sigementation problems.

    Parameters
    ----------
    n_labels: int
        Number of outputs in the segmentation model
    prefix: str
        Prefix to be displayed in the progressbar and tensorboard
    parse_output: Callable
        Function to parse the output
    parse_target: Callable
        Function to parse the target tensor
    cond: Callable
        Condition under which the metric will be updated
    """
    def __init__(self, n_labels=2, prefix="", parse_output=None,
                 parse_target=None, cond=None):
        super(MultilabelDiceMeter, self).__init__(name='dice', prefix=prefix, desc_name=None)
        self.__parse_output = parse_output
        self.__parse_target = parse_target if parse_target is not None else self.default_parse_target
        self.__cond = self.default_cond if cond is None else cond

        self.n_labels = n_labels

        self.values = np.zeros(n_labels)
        self.n_samples = 0

    def on_epoch_begin(self, *args, **kwargs):
        self.n_samples = 0
        self.values = np.zeros(self.n_labels)

    def on_minibatch_end(self, target, output, **kwargs):
        with torch.no_grad():
            if self.__cond(target, output):
                target = self.__parse_target(target)
                output = self.__parse_output(output)
                for i in range(target.size(0)):
                    t = target[i]
                    o = output[i]
                    coeffs = []
                    for cls in range(self.n_labels):
                        t_cls = t[cls]
                        o_cls = o[cls]
                        val = ItemWiseBinaryJaccardDiceMeter.compute_dice(t_cls.unsqueeze(0), o_cls.unsqueeze(0)).item()
                        coeffs.append(val)

                    self.n_samples += 1
                    self.values += np.array(coeffs)

    def current(self):
        return (self.values / self.n_samples).mean()
