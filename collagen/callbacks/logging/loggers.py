from abc import abstractmethod
from collections import OrderedDict
import torch.optim as optim
import numpy as np
import tqdm

from collagen.core import Callback
from collagen.core.utils import wrap_tuple

__all__ = ["Logger", "BatchLRLogger", "EpochLRLogger", "FakeScalarMeterLogger", "ProgressbarLogger", "ScalarMeterLogger"]


class Logger(Callback):
    def __init__(self, name=None):
        super().__init__(ctype="logger")
        self.__name = name

    @property
    def name(self):
        return self.__name

    @abstractmethod
    def on_batch_end(self, *args, **kwargs):
        raise NotImplementedError('Abstract class. Method is not implemented!')

    @abstractmethod
    def on_epoch_end(self, *args, **kwargs):
        raise NotImplementedError('Abstract class. Method is not implemented!')


class ScalarMeterLogger(Logger):
    def __init__(self, writer, log_dir: str = None, comment: str = '', name='scalar_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            if cb.ctype == "meter" and cb.current() is not None:
                self.__summary_writer.add_scalar(tag=cb.desc, scalar_value=cb.current(), global_step=epoch)


class FakeScalarMeterLogger(Logger):
    def __init__(self, writer, log_dir: str = None, comment: str = '', name='scalar_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        pass


class BatchLRLogger(Logger):
    def __init__(self, writer, optimizers, names, log_dir: str = None, tag: str = '', name='batch_lr_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__tag = tag
        self.__summary_writer = writer
        self.__optims = wrap_tuple(optimizers)
        self.__names = wrap_tuple(names)
        if len(self.__optims) != len(self.__names):
            raise ValueError(
                'The num of optimizers and names must match, but found {} and {}'.format(len(self.__optims),
                                                                                         len(self.__names)))
        self.__single = len(self.__names) == 1

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_end(self, epoch, batch_i, n_batches, *args, **kwargs):
        global_step = n_batches * epoch + batch_i
        if self.__single:
            assert len(self.__optims[0].param_groups) > 0, "Not found parameter in optimizer"
            self.__summary_writer.add_scalar(self.__tag, self.__optims[0].param_groups[0]['lr'], global_step)
        else:
            lrs = dict()
            for optim, name in zip(self.__optims, self.__names):
                lrs[name] = optim.param_groups[0]['lr']
            self.__summary_writer.add_scalars(self.__tag, lrs, global_step)


class EpochLRLogger(Logger):
    def __init__(self, writer, optimizers, names, log_dir: str = None, tag: str = '', name='epoch_lr_logger'):
        super().__init__(name=name)
        self.__log_dir = log_dir
        self.__tag = tag
        self.__summary_writer = writer
        self.__optims = wrap_tuple(optimizers)
        self.__names = wrap_tuple(names)
        if len(self.__optims) != len(self.__names):
            raise ValueError(
                'The num of optimizers and names must match, but found {} and {}'.format(len(self.__optims),
                                                                                         len(self.__names)))
        self.__single = len(self.__names) == 1

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, *args, **kwargs):
        if self.__single:
            assert len(self.__optims[0].param_groups) > 0, "Not found parameter in optimizer"
            self.__summary_writer.add_scalar(self.__tag, self.__optims[0].param_groups[0]['lr'], epoch)
        else:
            lrs = dict()
            for optim, name in zip(self.__optims, self.__names):
                lrs[name] = optim.param_groups[0]['lr']
            self.__summary_writer.add_scalars(self.__tag, lrs, epoch)


class ProgressbarLogger(Logger):
    def __init__(self, update_freq=1, optimizers=None, name='progressbar_logger', format_metric=None, format_lr=None):
        """Visualizes progressbar after a specific number of batches

        Parameters
        ----------
        update_freq: int
            The number of batches to update progressbar (default: 1)
        """
        super().__init__(name=name)
        self.__count = 0
        self.__update_freq = update_freq
        self.__optim = optimizers
        self.__format_metric = format_metric if format_metric is not None else self._default_format_metric
        self.__format_lr = format_lr if format_lr is not None else self._default_format_lr
        if not isinstance(self.__update_freq, int) or self.__update_freq < 1:
            raise ValueError(
                "`update_freq` must be `int` and greater than 0, but found {} {}".format(type(self.__update_freq),
                                                                                         self.__update_freq))

    def _check_freq(self):
        return self.__count % self.__update_freq == 0

    def on_epoch_end(self, *args, **kwargs):
        pass

    @staticmethod
    def _default_format_lr(x):
        return f"{x:0.1e}"

    @staticmethod
    def _default_format_metric(x):
        return f"{x:.03f}"

    @staticmethod
    def check_optims(opt):
        return not isinstance(opt, optim.Adam)

    def on_batch_end(self, strategy, epoch: int, progress_bar: tqdm, stage: str or None, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            if self.__optim is not None:
                if isinstance(self.__optim, optim.Optimizer) and self.check_optims(self.__optim):
                    postfix_progress['lr'] = self.__format_lr(self.__optim.param_groups[0]['lr'])
                elif isinstance(self.__optim, dict):
                    for opt_name in self.__optim:
                        if self.check_optims(self.__optim[opt_name]):
                            postfix_progress[f'lr{opt_name}'] = self.__format_lr(self.__optim[opt_name].param_groups[0]['lr'])
                elif isinstance(self.__optim, list) or isinstance(self.__optim, tuple):
                    for opt_i, opt in self.__optim:
                        if self.check_optims(opt):
                            postfix_progress[f'lr{opt_i}'] = self.__format_lr(opt.param_groups[0]['lr'])
                else:
                    raise TypeError(f'Not support optimizers type {type(self.__optim)}.')

            for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
                if cb.ctype == "meter" and cb.current() is not None:
                    list_metrics_desc.append(str(cb))
                    cb_cur = cb.current()
                    if not isinstance(cb_cur, dict):
                        if isinstance(cb_cur, np.ndarray):
                            cb_cur = cb_cur.item()
                        postfix_progress[cb.desc] = self.__format_metric(cb_cur)
                    else:
                        postfix_progress[cb.desc] = '|'.join([f'{cls}:{self.__format_metric(cb_cur[cls])}' for cls in cb_cur])
            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)