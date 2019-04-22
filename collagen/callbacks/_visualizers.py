import tqdm
import torch
import numpy as np
from torch import Tensor
from torch.tensor import OrderedDict
from typing import Tuple
from collagen.core import Callback
from collagen.strategies import Strategy
from torchvision.utils import make_grid
from collagen.core.utils import to_cpu
from collagen.metrics import plot_confusion_matrix


class ConfusionMatrixVisualizer(Callback):
    def __init__(self, writer, labels: list or None = None, tag="confusion_matrix", normalize=False):
        """ConfusionMatrixVisualizer class, which is a callback calculating accuracy after each forwarding step and
        exporting confusion matrix to TensorboardX at the end of each epoch

        Parameters
        ----------
        writer: TensorboardX SummaryWriter
            Writes confusion matrix figure into TensorboardX
        labels: list or None
            List of collected labels which are summarized in confusion matrix
        tag: str
            Tag of confusion matrix in TensorboardX
        normalize: bool
            If `True` display accurate percentage, otherwise, display accurate quantity
        """
        super().__init__(ctype="visualizer")
        self._labels = labels
        self._normalize = normalize
        self.__epoch = 0
        self._writer = writer
        self._tag = tag
        self._predicts = []
        self._corrects = []

    def _reset(self):
        self._predicts = []
        self._corrects = []

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__epoch = epoch
        self._reset()

    def on_forward_end(self, output: Tensor, target: Tensor, **kwargs):
        if len(target.shape) > 1 and target.shape[1] > 1:
            decoded_target_cls = target.argmax(dim=-1)
        elif len(target.shape) == 1:
            decoded_target_cls = target
        else:
            raise ValueError("target dims ({}) must be 1 or 2, but got {}".format(target.shape, len(target.shape)))

        if len(output.shape) > 1 and output.shape[1] > 1:
            decoded_pred_cls = output.argmax(dim=-1)
        elif len(output.shape) == 1:
            decoded_pred_cls = output
        else:
            raise ValueError("pred dims ({}) must be 1 or 2, but got {}".format(output.shape, len(output.shape)))

        self._corrects += [self._labels[i] for i in to_cpu(decoded_target_cls, use_numpy=True).tolist()]
        self._predicts += [self._labels[i] for i in to_cpu(decoded_pred_cls, use_numpy=True).tolist()]

    def on_epoch_end(self, *args, **kwargs):
        fig = plot_confusion_matrix(np.array(self._corrects), np.array(self._predicts), labels=self._labels, normalize=self._normalize)
        self._writer.add_figure(self._tag, fig, global_step=self.__epoch)


class ProgressbarVisualizer(Callback):
    def __init__(self, update_freq=1):
        """Visualizes progressbar after a specific number of batches

        Parameters
        ----------
        update_freq: int
            The number of batches to update progressbar (default: 1)
        """
        super().__init__(ctype="visualizer")
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
                if cb.ctype == "meter":
                    list_metrics_desc.append(str(cb))
                    postfix_progress[cb.desc] = f'{cb.current():.03f}'

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)


class TensorboardSynthesisVisualizer(Callback):
    def __init__(self, writer, generator_sampler, key_name: str = "data", tag: str = "Generated",
                 grid_shape: Tuple[int] or int = 10, split_channel=True, transform=None):
        """Visualizes synthesized images in TensorboardX

        Parameters
        ----------
        writer: TensorboardX SummaryWriter
            Writes metrics into TensorboardX
        generator_sampler: ItemLoader
            Loads item including synthesized image
        key_name: str
            Key corresponding to synthesized image in loaded samples from :attr:generator_sampler`
        tag: str
            Tag of metric in TensorboardX
        grid_shape: tuple of int or int
            Shape of the image grid. Will be generated according to the specified tuple - `WxH`. If int,
            the a square `grid_shape x grid_shape` will be generated
        split_channel: bool
            Whether split synthesized image by channels and concatenate them horizontally
        transform: function
            Transforms synthesized image
        """
        super().__init__(ctype="visualizer")
        self.__generator_sampler = generator_sampler
        self.__split_channel = split_channel

        if isinstance(grid_shape, tuple):
            if len(grid_shape) != 2:
                raise ValueError("`grid_shape` must have 2 dim, but found {}".format(len(grid_shape)))
        self.__transform = self._default_transform if transform is None else transform
        self.__writer = writer
        self.__key_name = key_name
        self.__grid_shape = grid_shape if isinstance(grid_shape, tuple) else (grid_shape, grid_shape)
        self.__num_images = self.__grid_shape[0] * self.__grid_shape[1]
        self.__num_batches = self.__num_images // self.__generator_sampler.batch_size + 1
        self.__tag = tag

    @staticmethod
    def _default_transform(x):
        return (x+1.0)/2.0

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        sampled_data = self.__generator_sampler.sample(self.__num_batches)
        images = []
        for i, dt in enumerate(sampled_data):
            if i < self.__num_images:
                for img in torch.unbind(dt[self.__key_name], dim=0):
                    img = self.__transform(img)
                    if len(img.shape) == 3 and img.shape[0] != 1 and img.shape[0] != 3:
                        if self.__split_channel:
                            if img.shape[0] % 3 == 0:
                                n_split = int(img.shape[0]/3)
                                separate_imgs = [img[3*k:3*(k+1), :, :] for k in range(n_split)]
                                concate_img = torch.cat(separate_imgs, dim=-1)
                                # print("concate_img0 shape: {}".format(concate_img.shape))
                                # concate_img = torch.unsqueeze(concate_img, 0)
                                # print("concate_img1 shape: {}".format(concate_img.shape))
                                images.append(concate_img)
                            else:
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
