import itertools
import re
from textwrap import wrap

import numpy as np
from matplotlib import figure
from sklearn.metrics import confusion_matrix

from typing import Tuple

import torch
import tqdm
from torch import Tensor
from torch.tensor import OrderedDict
from torchvision.utils import make_grid

from collagen.core import Callback
from collagen.core.utils import to_cpu


def plot_confusion_matrix(correct_labels, predict_labels, labels, normalize=True):
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 100.0 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = np.round(cm).astype('int')

    np.set_printoptions(precision=2)

    fig = figure.Figure(figsize=(5, 5), dpi=230, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=16)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=16, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=16)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=16,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig


class ConfusionMatrixVisualizer(Callback):
    def __init__(self, writer, labels: list or None = None, tag="confusion_matrix", normalize=True, name='cm',
                 cond=None, parse_class=None):
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
        self._name = name
        self.__cond = self._default_cond if cond is None else cond
        self.__parse_class = self._default_parse_class if parse_class is None else parse_class

    @staticmethod
    def _default_cond(target, output):
        return True

    @staticmethod
    def _default_parse_class(y):
        return y

    @property
    def targets(self):
        return self._corrects

    @property
    def predictions(self):
        return self._predicts

    @property
    def name(self):
        return self._name

    def _reset(self):
        self._predicts = []
        self._corrects = []

    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.__epoch = epoch
        self._reset()

    def on_forward_end(self, output: Tensor, target: Tensor or dict, **kwargs):
        if self.__cond(target, output):
            target_cls = self.__parse_class(target)
            pred_cls = self.__parse_class(output)
            if target_cls is not None and pred_cls is not None:
                # decoded_pred_cls = pred_cls.argmax(dim=-1)
                self._corrects += [self._labels[i] for i in to_cpu(target_cls, use_numpy=True).tolist()]
                self._predicts += [self._labels[i] for i in to_cpu(pred_cls, use_numpy=True).tolist()]

    def on_epoch_end(self, *args, **kwargs):
        if len(self._corrects) != len(self._predicts):
            raise ValueError(
                'Num of predictions and groundtruths must match, but found {} and {}'.format(len(self._predicts),
                                                                                             len(self._corrects)))
        elif len(self._corrects) > 0:
            fig = plot_confusion_matrix(np.array(self._corrects), np.array(self._predicts), labels=self._labels,
                                        normalize=self._normalize)
            self._writer.add_figure(self._tag, fig, global_step=self.__epoch)


class ProgressbarVisualizer(Callback):
    def __init__(self, update_freq=1, name='progressbar'):
        """Visualizes progressbar after a specific number of batches

        Parameters
        ----------
        update_freq: int
            The number of batches to update progressbar (default: 1)
        """
        super().__init__(ctype="visualizer")
        self.__count = 0
        self.__update_freq = update_freq
        self.__name = name
        if not isinstance(self.__update_freq, int) or self.__update_freq < 1:
            raise ValueError(
                "`update_freq` must be `int` and greater than 0, but found {} {}".format(type(self.__update_freq),
                                                                                         self.__update_freq))

    @property
    def name(self):
        return self.__name

    def _check_freq(self):
        return self.__count % self.__update_freq == 0

    def on_batch_end(self, strategy, epoch: int, progress_bar: tqdm, stage: str or None, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
                if cb.ctype == "meter" and cb.current() is not None:
                    list_metrics_desc.append(str(cb))
                    postfix_progress[cb.desc] = f'{cb.current():.03f}'

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)


class TensorboardSynthesisVisualizer(Callback):
    def __init__(self, writer, generator_sampler, key_name: str = "data", tag: str = "Generated",
                 grid_shape: Tuple[int] = (10, 10), split_channel=True, transform=None, unbind_imgs_transform=None):
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
        grid_shape: tuple
            Shape of synthesized image grip (default: (10, 10))
        split_channel: bool
            Whether split synthesized image by channels and concatenate them horizontally
        transform: function
            Transforms synthesized image
        """
        super().__init__(ctype="visualizer")
        self.__generator_sampler = generator_sampler
        self.__split_channel = split_channel

        if len(grid_shape) != 2:
            raise ValueError("`grid_shape` must have 2 dim, but found {}".format(len(grid_shape)))
        self.__transform = self._default_transform if transform is None else transform
        self.__writer = writer
        self.__key_name = key_name
        self.__grid_shape = grid_shape
        self.__num_images = grid_shape[0] * grid_shape[1]
        self.__num_batches = self.__num_images // self.__generator_sampler.batch_size + 1
        self.__tag = tag
        self.__unbind_imgs_transform = self._default_tranform_unbind_imgs if unbind_imgs_transform is None else unbind_imgs_transform

    @staticmethod
    def _default_transform(x):
        return (x + 1.0) / 2.0

    @staticmethod
    def _default_tranform_unbind_imgs(separate_imgs):
        concate_img = torch.cat(separate_imgs, dim=-1)
        return concate_img

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
                                n_split = int(img.shape[0] / 3)
                                separate_imgs = [img[3 * k:3 * (k + 1), :, :] for k in range(n_split)]
                                concate_img = self.__unbind_imgs_transform(separate_imgs)
                                images.append(concate_img)
                            else:
                                separate_imgs = torch.unbind(img, dim=0)
                                concate_img = self.__unbind_imgs_transform(separate_imgs)
                                images.append(torch.unsqueeze(concate_img, 0))
                        else:
                            raise ValueError(
                                "Channels of image ({}) must be either 1 or 3, but found {}".format(img.shape,
                                                                                                    img.shape[0]))
                    else:
                        images.append(img)
            else:
                break
        grid_images = make_grid(images[:self.__num_images], nrow=self.__grid_shape[0])
        self.__writer.add_images(self.__tag, img_tensor=grid_images, global_step=epoch, dataformats='CHW')
