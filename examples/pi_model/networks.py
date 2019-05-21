import torch.nn as nn
import torch.nn.functional as F
from torch.tensor import OrderedDict
from collagen.core import Module


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

        self.classify = nn.Sequential(nn.Conv2d(ndf * 8, n_cls, 4, 1, 0, bias=False),
                                      nn.Softmax(dim=1))  # state size. n_clsx1x1

        self.apply(weights_init)

    def get_features(self, x):
        f = self.main_flow(x)
        return f

    def forward(self, x):
        o3 = self.main_flow(x)
        classifier = self.classify(o3).squeeze(-1).squeeze(-1)
        return classifier

class Model01(Module):
    def __init__(self, nc=1, ndf=64, n_cls=10, ngpu=1, drop_rate=0.5):
        super().__init__()
        # input is (nc) x 32 x 32
        self.__ngpu = ngpu
        self.__drop_rate = drop_rate

        self._maxpool = nn.MaxPool2d(2, 2)

        self._dropout = nn.Dropout(p=self.__drop_rate)
        # input is (nc) x 64 x 64
        self._layer11 = nn.Sequential(nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer12 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer13 = nn.Sequential(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer21 = nn.Sequential(nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf*2),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer22 = nn.Sequential(nn.Conv2d(ndf*2, ndf*2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer23 = nn.Sequential(nn.Conv2d(ndf*2, ndf*2, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer31 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, 3, 1, 0, bias=False),
                                      nn.BatchNorm2d(ndf * 4),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer32 = nn.Sequential(nn.Conv2d(ndf*4, ndf*2, 1, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf * 2),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._layer33 = nn.Sequential(nn.Conv2d(ndf*2, ndf, 1, 1, 1, bias=False),
                                      nn.BatchNorm2d(ndf),
                                      nn.LeakyReLU(0.1, inplace=True))  # state size. (ndf) x 32 x 32

        self._gap = nn.AvgPool2d(6, 6)

        self._classifier = nn.Conv2d(128, n_cls, 1, 1, 0, bias=False)

        self.main_flow = nn.Sequential(self._layer11,
                                       self._layer12,
                                       self._layer13,

                                       self._maxpool,
                                       self._dropout,

                                       self._layer21,
                                       self._layer22,
                                       self._layer23,

                                       self._maxpool,
                                       self._dropout,

                                       self._layer31,
                                       self._layer32,
                                       self._layer33
                                       )

        self.apply(weights_init)

    def get_features(self, x):
        f = self.main_flow(x)
        f = self._gap(f).squeeze(-1).squeeze(-1)
        return f

    def forward(self, x):
        f = self.main_flow(x)
        f = self._gap(f)
        classifier = self._classifier(f).squeeze(-1).squeeze(-1)
        classifier = F.softmax(classifier, dim=-1)
        return classifier
