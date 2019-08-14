import argparse

import numpy as np
import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import torch
from torch import Tensor

from collagen.callbacks.logging.visualization import ConfusionMatrixVisualizer
from collagen.core.utils import to_cpu
from collagen.data.utils import ApplyTransform, Normalize, Compose


def cond_accuracy_meter(target, output):
    return target['name'].startswith('l')


def parse_target(target):
    if target['name'].startswith('l'):
        return target['target']
    else:
        return None


def parse_class(output):
    if isinstance(output, dict) and output['name'].startswith('l'):
        output = output['target']
    elif isinstance(output, Tensor):
        pass
    else:
        return None

    if output is None:
        return None
    elif len(output.shape) == 2:
        output_cpu = to_cpu(output.argmax(dim=1), use_numpy=True)
    elif len(output.shape) == 1:
        output_cpu = to_cpu(output, use_numpy=True)
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(output.shape)))
    output_cpu = output_cpu.astype(int)
    return output_cpu


class SSConfusionMatrixVisualizer(ConfusionMatrixVisualizer):
    def __init__(self, cond, parse_class, writer, labels: list or None = None, tag="confusion_matrix", normalize=False):
        super().__init__(writer=writer, labels=labels, tag=tag, normalize=normalize)
        self.__cond = cond
        self.__parse_class = parse_class

    def on_forward_end(self, output, target, **kwargs):
        if self.__cond(target, output):
            target_cls = self.__parse_class(target)
            pred_cls = self.__parse_class(output)
            if target_cls is not None and pred_cls is not None:
                # decoded_pred_cls = pred_cls.argmax(dim=-1)
                self._corrects += [self._labels[i] for i in to_cpu(target_cls, use_numpy=True).tolist()]
                self._predicts += [self._labels[i] for i in to_cpu(pred_cls, use_numpy=True).tolist()]


def wrap2solt(inp):
    if len(inp) == 2:
        img, label = inp
    else:
        img = inp
        label = -1
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img / 255.0, np.float32(target)


def init_transforms(nc=1):
    if nc == 1:
        norm_mean_std = Normalize((0.1307,), (0.3081,))
    elif nc == 3:
        norm_mean_std = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    else:
        raise ValueError("Not support channels of {}".format(nc))

    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.ResizeTransform(resize_to=(32, 32), interpolation='bilinear'),
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-10, 10), p=0.5),
            # slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=36),
            slt.CropTransform(crop_size=32, crop_mode='r'),
            slt.ImageAdditiveGaussianNoise(p=1.0)
        ]),
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    test_trf = Compose([
        wrap2solt,
        slt.ResizeTransform(resize_to=(32, 32), interpolation='bilinear'),
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    def custom_augment(img):
        tr = Compose([
            wrap2solt,
            slc.Stream([
                slt.ResizeTransform(resize_to=(32, 32), interpolation='bilinear'),
                slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
                slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
                slt.RandomRotate(rotation_range=(-10, 10), p=0.5),
                # slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
                slt.PadTransform(pad_to=36),
                slt.CropTransform(crop_size=32, crop_mode='r'),
                slt.ImageAdditiveGaussianNoise(p=1.0)
            ]),
            unpack_solt,
            ApplyTransform(norm_mean_std)
        ])

        img_tr, _ = tr((img, 0))
        return img_tr

    return train_trf, test_trf, custom_augment


def parse_item(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {'data': img, 'target': target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset name')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='Max learning rate')
    parser.add_argument('--initial_lr', default=0.0, type=float, help='Initial learning rate when using linear rampup')
    parser.add_argument('--lr_rampup', default=20, type=int, help='Length of learning rate rampup in the beginning')
    parser.add_argument('--lr_rampdown_epochs', default=80, type=int,
                        help='Length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--start_cycle_epoch', default=80, type=int, help='Epoch to start cycle')
    parser.add_argument('--cycle_rampdown_epochs', default=0, type=int, help='Length of epoch cycle to ramp down')
    parser.add_argument('--cycle_interval', default=20, type=int, help='Length of epoch for a cosine annealing cycle')

    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='Weight decay')
    parser.add_argument('--n_features', type=int, default=128, help='Number of features')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--grid_shape', type=tuple, default=(24, 24), help='Shape of grid of generated images')
    parser.add_argument('--n_training_batches', type=int, default=-1,
                        help='Num of training batches, if -1, auto computed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args
