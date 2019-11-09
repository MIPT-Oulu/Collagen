from abc import abstractmethod
from collections import OrderedDict

import tqdm

from collagen.core import Callback
from collagen.core.utils import wrap_tuple


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
    def __init__(self, update_freq=1, name='progressbar_logger'):
        """Visualizes progressbar after a specific number of batches

        Parameters
        ----------
        update_freq: int
            The number of batches to update progressbar (default: 1)
        """
        super().__init__(name=name)
        self.__count = 0
        self.__update_freq = update_freq
        if not isinstance(self.__update_freq, int) or self.__update_freq < 1:
            raise ValueError(
                "`update_freq` must be `int` and greater than 0, but found {} {}".format(type(self.__update_freq),
                                                                                         self.__update_freq))

    def _check_freq(self):
        return self.__count % self.__update_freq == 0

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_batch_end(self, strategy, epoch: int, progress_bar: tqdm, stage: str or None, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
                if cb.ctype == "meter" and cb.current() is not None:
                    list_metrics_desc.append(str(cb))
                    cb_cur = cb.current()
                    if not isinstance(cb_cur, dict):
                        postfix_progress[cb.desc] = f'{cb_cur:.03f}'
                    else:
                        postfix_progress[cb.desc] = '|'.join([f'{cls}:{cb_cur[cls]:.03f}' for cls in cb_cur])

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)