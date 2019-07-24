from collagen.core import Callback
from collagen.core.utils import wrap_tuple


class Logging(Callback):
    def __init__(self):
        super().__init__(ctype="logger")
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass


class MeterLogging(Logging):
    def __init__(self, writer, log_dir: str = None, comment: str = ''):
        super().__init__()
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            if cb.ctype == "meter" and cb.current() is not None:
                self.__summary_writer.add_scalar(tag=cb.desc, scalar_value=cb.current(), global_step=epoch)


class BatchLRLogging(Logging):
    def __init__(self, writer, optimizers, names, log_dir: str = None, tag: str = ''):
        super().__init__()
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


class EpochLRLogging(Logging):
    def __init__(self, writer, optimizers, names, log_dir: str = None, tag: str = ''):
        super().__init__()
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

    def on_epoch_end(self, epoch, *args, **kwargs):
        if self.__single:
            assert len(self.__optims[0].param_groups) > 0, "Not found parameter in optimizer"
            self.__summary_writer.add_scalar(self.__tag, self.__optims[0].param_groups[0]['lr'], epoch)
        else:
            lrs = dict()
            for optim, name in zip(self.__optims, self.__names):
                lrs[name] = optim.param_groups[0]['lr']
            self.__summary_writer.add_scalars(self.__tag, lrs, epoch)
