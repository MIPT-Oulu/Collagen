from collagen.callbacks import Callback


class Logging(Callback):
    def __init__(self):
        self.__type = "logger"
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

    def on_epoch_end(self, epoch, strategy, **kwargs):
        for cb in strategy.get_callbacks_by_name("minibatch"):
            if cb.get_type() == "meter":
                self.__summary_writer.add_scalar(tag=cb.get_name(), scalar_value=cb.current(), global_step=epoch)
