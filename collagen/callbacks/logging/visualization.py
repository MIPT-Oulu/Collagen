import itertools
import re
import random
from textwrap import wrap

import numpy as np
from matplotlib import figure
from sklearn.metrics import confusion_matrix

from typing import Tuple

import torch
from torch import Tensor
from torchvision.utils import make_grid

from collagen.core import Callback
from collagen.core.utils import to_cpu


class ConfusionMatrixVisualizer(Callback):
    def __init__(self, writer, labels: list or None = None, tag="confusion_matrix", normalize=True, name='cm',
                 cond=None, parse_class=None, parse_output=None, parse_target=None):
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
        if parse_target is None or parse_output is None:
            self.__parse_class = self._default_parse_class if parse_class is None else parse_class
            self.__parse_target = self.__parse_class if parse_target is None else parse_target
            self.__parse_output = self.__parse_class if parse_output is None else parse_output
        else:
            self.__parse_target = parse_target
            self.__parse_output = parse_output

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
            target_cls = self.__parse_target(target)
            pred_cls = self.__parse_output(output)
            if target_cls is not None and pred_cls is not None:
                # decoded_pred_cls = pred_cls.argmax(dim=-1)
                self._corrects += [self._labels[i] for i in to_cpu(target_cls, use_numpy=True).tolist()]
                self._predicts += [self._labels[i] for i in to_cpu(pred_cls, use_numpy=True).tolist()]

    @staticmethod
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

    def on_epoch_end(self, *args, **kwargs):
        if len(self._corrects) != len(self._predicts):
            raise ValueError(
                'Num of predictions and groundtruths must match, but found {} and {}'.format(len(self._predicts),
                                                                                             len(self._corrects)))
        elif len(self._corrects) > 0:
            fig = self.plot_confusion_matrix(np.array(self._corrects), np.array(self._predicts),
                                             labels=self._labels,
                                             normalize=self._normalize)
            self._writer.add_figure(self._tag, fig, global_step=self.__epoch)


class ImageSamplingVisualizer(Callback):
    def __init__(self, writer, generator_sampler, key_name: str = "data", tag: str = "Generated",
                 grid_shape: Tuple[int, int] or int = (10, 10), split_channel=True, transform=None, unbind_imgs_transform=None):
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
        self.__unbind_imgs_transform = self._default_transform_unbind_imgs if unbind_imgs_transform is None else unbind_imgs_transform

    @staticmethod
    def _default_transform(x):
        return (x + 1.0) / 2.0

    @staticmethod
    def _default_transform_unbind_imgs(separate_imgs):
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
                                concat_img = self.__unbind_imgs_transform(separate_imgs)
                                images.append(concat_img)
                            else:
                                separate_imgs = torch.unbind(img, dim=0)
                                concat_img = self.__unbind_imgs_transform(separate_imgs)
                                images.append(torch.unsqueeze(concat_img, 0))
                        else:
                            raise ValueError(
                                "Channels of image ({}) must be either 1 or 3, but found {}".format(img.shape,
                                                                                                    img.shape[0]))
                    else:
                        images.append(img)
            else:
                break
        grid_images = make_grid(images[:self.__num_images], nrow=self.__grid_shape[0])
        self.__writer.add_images(self.__tag, img_tensor=grid_images,  dataformats='CHW')


class FakeImageSamplingVisualizer(Callback):
    def __init__(self, writer, generator_sampler, key_name: str = "data", tag: str = "Generated",
                 grid_shape: Tuple[int, int] or int = (10, 10), split_channel=True, transform=None,
                 unbind_imgs_transform=None):
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
        self.__unbind_imgs_transform = self._default_transform_unbind_imgs if unbind_imgs_transform is None else unbind_imgs_transform

    @staticmethod
    def _default_transform(x):
        return (x + 1.0) / 2.0

    @staticmethod
    def _default_transform_unbind_imgs(separate_imgs):
        concate_img = torch.cat(separate_imgs, dim=-1)
        return concate_img

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        sampled_data = self.__generator_sampler.sample(self.__num_batches)
        pass



class ImageMaskVisualizer(Callback):
    """
    Expected behavior:

    0. specify the comparator
    1. tracks the metric
    2. tracks the worst and the best minibatches on CPU. If the minibatch metric is < than the best minibatch -> randomly sample image and mask.
    3. Do the same for the worst cases
    4. Visualize: image, and image with mask
    5. ensure support for binary and multi-class semantica segmentation.

    """
    def __init__(self, writer, log_dir: str = None, comment: str = '', grid_shape: int = (2, 1),
                 mean: float = None, std: float = None):
        super().__init__(ctype="visualizer")
        self.__log_dir = log_dir
        self.__comment = comment
        self.__summary_writer = writer
        self.__num_batches = 2
        self.__grid_shape = grid_shape
        self.__num_images = grid_shape[0] * grid_shape[1]
        self.__min_loss = None
        self.__max_loss = None
        self.__mean = mean
        self.__std = std
        self.__input_best = None
        self.__input_worst = None
        self.__pred_best = None
        self.__pred_worst = None

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_minibatch_end(self, loss, session, batches_count, input, output, **kwargs):
        if hasattr(session.loss, 'get_loss_by_name'):
            metric_value = session.loss.get_loss_by_name(self.name)
        else:
            metric_value = loss
        # Update metric
        if metric_value is not None:
            # No previous metric
            if self.__min_loss is None:
                self.__min_loss = metric_value
                self.__max_loss = metric_value
                self.__input_worst = input.detach()
                self.__pred_worst = output.detach()
                self.__input_best = input.detach()
                self.__pred_best = output.detach()
            # Best case improved
            elif metric_value < self.__min_loss:
                self.__min_loss = metric_value
                self.__input_best = input.detach()
                self.__pred_best = output.detach()
            # Worst case decreased
            elif metric_value > self.__max_loss:
                self.__max_loss = metric_value
                self.__input_worst = input.detach()
                self.__pred_worst = output.detach()

    def on_epoch_begin(self, *args, **kwargs):
        # Reset the variables for new epoch
        self.__min_loss, self.__max_loss = None, None
        self.__input_best, self.__pred_best = None, None
        self.__input_worst, self.__pred_worst = None, None

    def on_epoch_end(self, epoch, strategy, stage, **kwargs):
        # Plot images on tensorboard
        self.__summary_writer.add_images(tag='Input_best', dataformats='NCHW', global_step=epoch,
                                         img_tensor=self.undo_transform(self.__input_best, self.__mean, self.__std))

        self.__summary_writer.add_images(tag='Prediction_best', img_tensor=torch.sigmoid(self.__pred_best),
                                         dataformats='NCHW', global_step=epoch)

        self.__summary_writer.add_images(tag='Input_worst', dataformats='NCHW', global_step=epoch,
                                         img_tensor=self.undo_transform(self.__input_worst, self.__mean, self.__std))

        self.__summary_writer.add_images(tag='Prediction_worst', img_tensor=torch.sigmoid(self.__pred_worst),
                                         dataformats='NCHW', global_step=epoch)


    @staticmethod
    def undo_transform(data, mean, std):
        # img is CxHxW tensor
        # batch dimension should be absent
        data = data.clone()
        if len(data.shape) == 4:
            for im in range(data.size(0)):
                for c_ind in range(data.size(1)):
                    data[im, c_ind] = data[im, c_ind]*std[c_ind] + mean[c_ind]
        else:
            for c_ind in range(data.size(0)):
                data[c_ind] = data[c_ind] * std[c_ind] + mean[c_ind]
        return data