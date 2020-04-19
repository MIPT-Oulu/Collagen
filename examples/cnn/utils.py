import argparse

import torch.nn.functional as F
from torch import nn
import numpy as np

import solt
import solt.transforms as slt

from collagen.core import Module


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bw', type=int, default=64, help='Bandwidth of model')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--snapshots', default='snapshots', help='Where to save the snapshots')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset name')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    args = parser.parse_args()

    return args


def parse_item_mnist(root, entry, trf, data_key, target_key):
    img = entry[data_key]

    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    if img.shape[-1] == 1:
        stats = {'mean': (0.1307,), 'std': (0.3081,)}
    elif img.shape[-1] == 3:
        stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.261)}
    else:
        raise ValueError("Not support channels of {}".format(img.shape[-1]))

    trf_data = trf({'image': img}, normalize=True, **stats)
    return {data_key: trf_data['image'], target_key: entry[target_key]}


def init_mnist_transforms():
    train_trf = solt.Stream([
        slt.Scale(range_x=(0.9, 1.1), same=False, p=0.5),
        slt.Shear(range_x=(-0.05, 0.05), p=0.5),
        slt.Rotate((-5, 5), p=0.5),
        slt.Pad(pad_to=(32, 32))
    ])

    test_trf = solt.Stream([slt.Pad(pad_to=(32, 32))])

    return train_trf, test_trf


class SimpleConvNet(Module):
    def __init__(self, bw, drop=0.5, n_cls=10, n_channels=1):
        super().__init__()
        self.n_filters_last = bw * 2

        self.conv1 = self.make_layer(n_channels, bw)
        self.conv2 = self.make_layer(bw, bw)
        self.conv3 = self.make_layer(bw, bw * 2)
        self.conv4 = self.make_layer(bw * 2, self.n_filters_last)

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(nn.Dropout(drop),
                                        nn.Linear(self.n_filters_last, n_cls))

    @staticmethod
    def make_layer(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 3, 1, 1),
                             nn.BatchNorm2d(out),
                             nn.ReLU(True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # 16x16
        x = self.conv3(x)
        x = self.pool(x)  # 8x8
        x = self.conv4(x)
        x = self.pool(x)  # 4x4

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(x.size(0), -1)

        return self.classifier(x)
