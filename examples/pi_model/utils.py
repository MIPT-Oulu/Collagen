import argparse
from collagen.core import Module
from torch import nn
from torch.tensor import OrderedDict
import torch
from torch import Tensor
import numpy as np
import solt.data as sld
from collagen.core.utils import to_cpu
from collagen.data.utils import ApplyTransform, Normalize, Compose
from collagen.metrics import AccuracyMeter
from collagen.callbacks import ConfusionMatrixVisualizer
import solt.core as slc
import solt.transforms as slt

class SSConfusionMatrixVisualizer(ConfusionMatrixVisualizer):
    def __init__(self, writer, labels: list or None = None, tag="confusion_matrix"):
        super().__init__(writer=writer, labels=labels, tag=tag)

    def on_forward_end(self, output, target, **kwargs):
        if isinstance(target, Tensor):
            target_cls = target.type(torch.int64)
        elif isinstance(target, tuple):
            target_cls = target[0].type(torch.int64)
        pred_cls = output
        decoded_pred_cls = pred_cls.argmax(dim=-1)
        # decoded_target_cls = target_cls.argmax(dim=-1)
        self._corrects += [self._labels[i] for i in to_cpu(target_cls, use_numpy=True).tolist()]
        self._predicts += [self._labels[i] for i in to_cpu(decoded_pred_cls, use_numpy=True).tolist()]

def cond_accuracy_meter(target, output):
    return True

def parse_target_accuracy_meter(target):
    if isinstance(target, Tensor):
        return target
    elif isinstance(target, tuple):
        return target[0]
    else:
        raise ValueError("Invalid target!")


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
    return img/255.0, np.float32(target)


def init_transforms(nc=1):
    if nc == 1:
        norm_mean_std = Normalize((0.5,), (0.5,))
    elif nc == 3:
        norm_mean_std = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    augment = Compose([
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

    return train_trf, test_trf, augment


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(Module):
    def __init__(self, nc=1, ndf=64, n_cls=10, ngpu=1, drop_rate=0.35):
        super(Discriminator, self).__init__()
        # input is (nc) x 32 x 32
        self.__ngpu = ngpu
        self.__drop_rate = drop_rate

        self.dropout = nn.Dropout(p=self.__drop_rate)
        # input is (nc) x 64 x 64
        self._layer1 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 2),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*2) x 16 x 16

        self._layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 4),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*4) x 8 x 8

        self._layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ndf * 8),
                                     nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*4) x 4 x 4

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    # ("dropout1", self.dropout),
                                                    ("conv_block2", self._layer2),
                                                    # ("dropout2", self.dropout),
                                                    ("conv_block3", self._layer3),
                                                    # ("dropout3", self.dropout),
                                                    ("conv_block4", self._layer4),
                                                    # ("dropout3", self.dropout),
                                                    # ("conv_final", self._layer5)
                                                    ]))

        self.classify = nn.Sequential(nn.Conv2d(ndf * 8, n_cls + 1, 4, 1, 0, bias=False),
                                      nn.Softmax(dim=1))  # state size. n_clsx1x1

        self.apply(weights_init)

    def get_features(self, x):
        f = self.main_flow(x)
        return f

    def forward(self, x):
        o3 = self.main_flow(x)
        classifier = self.classify(o3).squeeze(-1).squeeze(-1)
        return classifier


def parse_item(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {'data': img, 'target': target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (Discriminator)')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='Weight decay')
    parser.add_argument('--n_features', type=int, default=128, help='Number of features')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--grid_shape', type=tuple, default=(24, 24), help='Shape of grid of generated images')
    parser.add_argument('--ngpu', type=int, default=1, help='Num of GPUs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args




