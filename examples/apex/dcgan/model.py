from collections import OrderedDict

import torch
from torch import nn

from collagen.core import Module


class Discriminator(Module):
    def __init__(self, nc=1, ndf=64, n_gpu=1):
        super(Discriminator, self).__init__()
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

        # self._layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #                              nn.Sigmoid())  # state size. 1x1x1
        self._layer5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        self.main_flow = nn.Sequential(OrderedDict([("conv_block1", self._layer1),
                                                    ("conv_block2", self._layer2),
                                                    ("conv_block3", self._layer3),
                                                    ("conv_block4", self._layer4),
                                                    ("conv_final", self._layer5)
                                                    ]))

        self.main_flow.apply(weights_init)

    def forward(self, x: torch.tensor):
        output = self.main_flow(x)
        return output.view(-1, 1).squeeze(1)

    def get_features(self):
        pass

    def get_features_by_name(self, name):
        pass


class Generator(Module):
    def __init__(self, nc=1, nz=100, ngf=64, n_gpu=1):
        super(Generator, self).__init__()

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

        output = self.main_flow(x)
        return output

    def get_features(self):
        pass

    def get_features_by_name(self, name):
        pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
