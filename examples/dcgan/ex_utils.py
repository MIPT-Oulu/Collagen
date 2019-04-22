import argparse
from collagen.core import Module
from torch import nn
from torch.tensor import OrderedDict
import torch
import numpy as np
import solt.data as sld
from collagen.data.utils import ApplyTransform, Normalize, Compose
import solt.transforms as slt
import random


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img/255.0, np.float32(target)


def init_mnist_transforms():
    return Compose([
        wrap2solt,
        slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(Module):
    def __init__(self, nc=1, ndf=64, n_gpu=1):
        super(Discriminator, self).__init__()

        self.__devices = list(range(n_gpu))

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

        self._layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                                     nn.Sigmoid())  # state size. 1x1x1

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    ("conv_block2", self._layer2),
                                                    ("conv_block3", self._layer3),
                                                    ("conv_block4", self._layer4),
                                                    ("conv_final", self._layer5)
                                                    ]))

        self.main_flow.apply(weights_init)

    def forward(self, x: torch.tensor):
        if x.is_cuda and len(self.__devices) > 1:
            output = nn.parallel.data_parallel(self.main_flow, x, self.__devices)
        else:
            output = self.main_flow(x)

        return output.view(-1, 1).squeeze(1)


class Generator(Module):
    def __init__(self, nc=1, nz=100, ngf=64, n_gpu=1):
        super(Generator, self).__init__()

        self.__devices = list(range(n_gpu))

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

        self._layer5 = nn.Sequential(nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(ngf // 2),
                                     nn.ReLU(True))  # state size. (ngf) x 64 x 64

        self._layer6 = nn.Sequential(nn.Conv2d(ngf // 2, nc, 3, 1, 1, bias=False),
                                     nn.Tanh())  # state size. (ngf) x 64 x 64

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    ("conv_block2", self._layer2),
                                                    ("conv_block3", self._layer3),
                                                    ("conv_block4", self._layer4),
                                                    ("conv_block5", self._layer5),
                                                    ("conv_final", self._layer6)
                                                    ]))

        self.main_flow.apply(weights_init)

    def forward(self, x: torch.tensor):
        if len(x.size()) != 2:
            raise ValueError("Input must have 2 dim but found {}".format(x.shape))
        x = x.view(x.size(0), x.size(1), 1, 1)

        if x.is_cuda and len(self.__devices) > 1:
            output = nn.parallel.data_parallel(self.main_flow, x, device_ids=self._devices)
        else:
            output = self.main_flow(x)

        return output


def parse_item_mnist_gan(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'data': img, 'target': np.float32(1.0), 'class': target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=64, help='Latent space size')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--ngpu', type=int, default=1, help='Num of GPUs')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="dcgan", help='Comment of log')
    parser.add_argument('--grid_shape', type=int, default=8, help='Shape of grid of generated images')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return args
