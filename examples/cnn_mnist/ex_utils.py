import argparse
import torch
import numpy as np

import pandas as pd
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch import nn

import solt.data as sld
import solt.core as slc
import solt.transforms as slt

from collagen.data.utils import ApplyTransform, Normalize, Compose
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
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    args = parser.parse_args()

    return args


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img, target


def get_mnist(data_folder='.', train=True):
    mnist_db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    list_rows = [{"img": np.array(item), "target": target.item()} for item, target in mnist_db]
    meta_data = pd.DataFrame(list_rows)

    return meta_data, list(range(10))


def init_mnist_transforms():
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
        ApplyTransform(Normalize((0.1307,), (0.3081,)))
    ])

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(Normalize((0.1307,), (0.3081,)))
    ])

    return train_trf, test_trf


class SimpleConvNet(Module):
    def __init__(self, bw, drop=0.5, n_cls=10):
        super(SimpleConvNet, self).__init__()
        self.n_filters_last = bw*2

        self.conv1 = self.make_layer(1, bw)
        self.conv2 = self.make_layer(bw, bw*2)
        self.conv3 = self.make_layer(bw*2, self.n_filters_last)

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

