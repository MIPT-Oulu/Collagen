import torch
from torch import nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            self.branch1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False))
        else:
            self.branch1 = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return x1 + x2


class ResNet(nn.Module):
    def __init__(self, in_channels=3, n_features=64, drop_rate=0.2, n_classes=10):
        super().__init__()
        c = [n_features, 2 * n_features, 4 * n_features, 4 * n_features]
        self.prep = nn.Conv2d(in_channels, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = nn.Sequential(ResBlock(c[0], c[0], 1),
                                    ResBlock(c[0], c[0], 1))

        self.layer2 = nn.Sequential(ResBlock(c[0], c[1], 2),
                                    ResBlock(c[1], c[1], 1))

        self.layer3 = nn.Sequential(ResBlock(c[1], c[2], 2),
                                    ResBlock(c[2], c[2], 1))

        self.layer4 = nn.Sequential(ResBlock(c[2], c[3], 2),
                                    ResBlock(c[3], c[3], 1))

        self.drop = nn.Dropout(drop_rate)

        self.avgpool = nn.AvgPool2d(4)
        self.maxpool = nn.MaxPool2d(4)
        self.linear = nn.Linear(c[3], n_classes, bias=True)


    def forward(self, x):
        o = self.prep(x)
        o = self.layer1(o)
        o = self.drop(o)
        o = self.layer2(o)
        o = self.drop(o)
        o = self.layer3(o)
        o = self.drop(o)
        o = self.layer4(o)
        o = self.avgpool(o).view(o.shape[0], -1)
        o = self.drop(o)
        out = self.linear(o)

        # o1 = self.avgpool(o)
        # o2 = self.maxpool(o)
        # o12 = torch.cat((o1, o2), dim=1).view(o1.shape[0], -1)
        # o12 = self.drop(o12)
        # out = self.linear(o12)

        return out