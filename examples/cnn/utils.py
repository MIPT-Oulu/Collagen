import argparse

import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import torch
import torch.nn.functional as F
from torch import nn

from collagen.core import Module
from collagen.data.utils import ApplyTransform, Normalize, Compose


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bw', type=int, default=64, help='Bandwidth of model')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--snapshots', default='snapshots', help='Where to save the snapshots')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset name')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    parser.add_argument('--distributed', type=bool, default=True, help='whether to use DDP')
    parser.add_argument('--gpu', type=int, default=0, help='Default GPU id')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()

    return args


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img, target


def init_mnist_cifar_transforms(n_channels=1, stage='train'):
    if n_channels == 1:
        norm_mean_std = Normalize((0.1307,), (0.3081,))
    elif n_channels == 3:
        norm_mean_std = Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    else:
        raise ValueError("Not support channels of {}".format(n_channels))

    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=34),
            slt.CropTransform(crop_size=32, crop_mode='r')
        ]),
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    if stage == 'train':
        return train_trf

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(norm_mean_std)
    ])

    return test_trf


class SimpleConvNet(Module):
    def __init__(self, bw, drop=0.5, n_cls=10, n_channels=1):
        super(SimpleConvNet, self).__init__()
        self.n_filters_last = bw * 2

        self.conv1 = self.make_layer(n_channels, bw)
        self.conv2 = self.make_layer(bw, bw * 2)
        self.conv3 = self.make_layer(bw * 2, self.n_filters_last)

        self.classifier = nn.Sequential(nn.Dropout(drop),
                                        nn.Linear(self.n_filters_last, n_cls))

    @staticmethod
    def make_layer(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 3, 1, 1),
                             nn.BatchNorm2d(out),
                             nn.ReLU(True))

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)  # 16x16
        x = F.max_pool2d(self.conv2(x), 2)  # 8x8
        x = F.max_pool2d(self.conv3(x), 2)  # 4x4

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(x.size(0), -1)

        return self.classifier(x)

    def get_features(self):
        pass

    def get_features_by_name(self, name):
        pass
