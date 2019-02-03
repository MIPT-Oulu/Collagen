import tqdm
import torch
from torch.tensor import OrderedDict
from typing import Tuple
from collagen.core import Callback
from collagen.strategies import Strategy
from torchvision.utils import make_grid


class ProgressbarVisualizer(Callback):
    def __init__(self, update_freq=1):
        super().__init__(type="visualizer")
        self.__count = 0
        self.__update_freq = update_freq
        if not isinstance(self.__update_freq, int) or self.__update_freq < 1:
            raise ValueError(
                "`update_freq` must be `int` and greater than 0, but found {} {}".format(type(self.__update_freq),
                                                                                         self.__update_freq))

    def _check_freq(self):
        return self.__count % self.__update_freq == 0

    def on_batch_end(self, strategy: Strategy, epoch: int, progress_bar: tqdm, stage: str or None, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
                if cb.get_type() == "meter":
                    list_metrics_desc.append(str(cb))
                    postfix_progress[cb.get_name()] = f'{cb.current():.03f}'

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)


class TensorboardSynthesisVisualizer(Callback):
    def __init__(self, writer, generator_sampler, key_name: str = "data", tag: str = "Generated",
                 grid_shape: Tuple[int] = (10, 10), split_channel=True):
        super().__init__(type="visualizer")
        self.__generator_sampler = generator_sampler
        self.__split_channel = split_channel

        if len(grid_shape) != 2:
            raise ValueError("`grid_shape` must have 2 dim, but found {}".format(len(grid_shape)))

        self.__writer = writer
        self.__key_name = key_name
        self.__grid_shape = grid_shape
        self.__num_images = grid_shape[0] * grid_shape[1]
        self.__num_batches = self.__num_images // self.__generator_sampler.batch_size + 1
        self.__tag = tag

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        sampled_data = self.__generator_sampler.sample(self.__num_batches)
        images = []
        for i, dt in enumerate(sampled_data):
            if i < self.__num_images:
                for img in torch.unbind(dt[self.__key_name], dim=0):
                    if len(img.shape) == 3 and img.shape[0] != 1 and img.shape[0] != 3:
                        if self.__split_channel:
                            separate_imgs = torch.unbind(img, dim=0)
                            concate_img = torch.cat(separate_imgs, dim=-1)
                            images.append(torch.unsqueeze(concate_img, 0))
                        else:
                            raise ValueError("Channels of image ({}) must be either 1 or 3, but found {}".format(img.shape, img.shape[0]))
                    else:
                        images.append(img)
            else:
                break
        grid_images = make_grid(images[:self.__num_images], nrow=self.__grid_shape[0])
        self.__writer.add_images(self.__tag, img_tensor=grid_images, global_step=epoch, dataformats='CHW')
