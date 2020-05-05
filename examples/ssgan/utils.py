import argparse

import numpy as np
import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import torch
from torch import nn
from torch.tensor import OrderedDict

from collagen.core import Module
from collagen.data.utils import ApplyTransform, Normalize, Compose


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img / 255.0, np.float32(target)


def init_mnist_transforms():
    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-10, 10), p=0.5),
            # slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=70),
            slt.CropTransform(crop_size=64, crop_mode='r'),
            slt.ImageAdditiveGaussianNoise(p=1.0)
        ]),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])

    test_trf = Compose([
        wrap2solt,
        slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
        # slt.PadTransform(pad_to=64),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,))),

    ])

    return train_trf, test_trf


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
                                                    ("dropout1", self.dropout),
                                                    ("conv_block2", self._layer2),
                                                    ("dropout2", self.dropout),
                                                    ("conv_block3", self._layer3),
                                                    ("dropout3", self.dropout),
                                                    ("conv_block4", self._layer4),
                                                    ("dropout3", self.dropout),
                                                    # ("conv_final", self._layer5)
                                                    ]))

        self.valid = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                   nn.Sigmoid())  # state size. 1x1x1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 8, n_cls + 1, 4, 1, 0, bias=False),
                                      nn.Softmax(dim=1))  # state size. n_clsx1x1

        self.apply(weights_init)

    def forward(self, x):
        o3 = self.main_flow(x)
        validator = self.valid(o3).squeeze(-1).squeeze(-1)
        classifier = self.classify(o3).squeeze(-1).squeeze(-1)
        return torch.cat((classifier, validator), dim=-1)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64, ngpu=1):
        super(Generator, self).__init__()
        self.__ngpu = ngpu

        self._layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                     nn.BatchNorm2d(ngf * 8),
                                     nn.ReLU(True))  # state size. (ngf*8) x 4 x 4

        self._layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf * 4),
                                     nn.ReLU(True))  # state size. (ngf*2) x 8 x 8

        self._layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf * 2),
                                     nn.ReLU(True))  # state size. (ngf*2) x 16 x 16

        self._layer4 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf),
                                     nn.ReLU(True))  # state size. (ngf) x 32 x 32

        self._layer5 = nn.Sequential(nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
                                     nn.Tanh())  # state size. (nc) x 64 x 64

        # self._layer6 = nn.Sequential(nn.Conv2d(ngf // 2, 1, 3, 1, 1, bias=False),
        #                              nn.Sigmoid())  # state size. (ngf) x 64 x 64

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    ("conv_block2", self._layer2),
                                                    ("conv_block3", self._layer3),
                                                    ("conv_block4", self._layer4),
                                                    ("conv_block5", self._layer5),
                                                    # ("conv_final", self._layer6)
                                                    ]))

        self.main_flow.apply(weights_init)

    def forward(self, x):
        if len(x.size()) != 2:
            raise ValueError("Input must have 2 dim but found {}".format(x.shape))
        x = x.view(x.size(0), x.size(1), 1, 1)

        if x.is_cuda and self.__ngpu > 1:
            output = nn.parallel.data_parallel(self.main_flow, x, range(self.__ngpu))
        else:
            output = self.main_flow(x)

        return output


def parse_item_mnist_ssgan(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    ext_y = np.zeros(10 + 2, dtype=np.float32)
    ext_y[-1] = 1.0
    ext_y[int(round(target))] = 1.0
    return {data_key: img, target_key: ext_y, 'class': target, 'valid': ext_y[-1]}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--d_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--beta1', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--grid_shape', type=tuple, default=(24, 24), help='Shape of grid of generated images')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args
