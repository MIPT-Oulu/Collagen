import argparse

import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import torch
import torch.nn.functional as F
from torch import nn

from collagen.core import Module
from collagen.core.utils._utils import find_free_localhost_port
from collagen.data.utils import ApplyTransform, Normalize, Compose


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



if __name__ == '__main__':
    print(find_free_localhost_port())