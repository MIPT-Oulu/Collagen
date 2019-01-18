from tensorboardX import SummaryWriter

from collagen.core._callback import Callback
from typing import Tuple


class Logging(Callback):
    def __init__(self):
        self.__type = "logger"
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass


class MeterLogging(Logging):
    def __init__(self, log_dir: str = None, comment: str = ''):
        super().__init__()
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = SummaryWriter(log_dir=self.__log_dir, comment=self.__comment)

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, callbacks: Tuple[Callback], epoch, **kwargs):
        for cb in callbacks:
            if cb.get_type() == "meter":
                self.__summary_writer.add_scalar(tag=cb.get_name(), scalar_value=cb.current(), global_step=epoch)


# class GANLogging(Logging):
#     def __init__(self, log_fake_images: bool = True, image_tile_shape: Tuple[int] = (6,6), std_image_shape: Tuple[int] = (64,64)):
#         super(GANLogging, self).__init__()
#         self.__log_fake_images = log_fake_images
#         self.__image_tile_shape = image_tile_shape
#         self.__std_image_shape = std_image_shape
#
#     def on_epoch_end(self, *args, **kwargs):
