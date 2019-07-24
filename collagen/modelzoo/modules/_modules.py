from torch import nn
from torch.nn import functional as F
from collagen.core import Module


class ConvBlock(Module):
    def __init__(self, ks, inp, out, stride=1, pad=1, activation='relu',
                 normalization='BN', bias=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(inp, out, kernel_size=ks, padding=pad, stride=stride, bias=bias), ]

        if normalization == 'BN':
            layers.append(nn.BatchNorm2d(out))
        elif normalization == 'IN':
            layers.append(nn.InstanceNorm2d(out))
        elif normalization is None:
            pass
        else:
            raise NotImplementedError('Not supported normalization type!')

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'selu':
            layers.append(nn.SELU(inplace=True))
        elif activation == 'elu':
            layers.append(nn.ELU(1, inplace=True))
        elif activation is None:
            pass
        else:
            raise NotImplementedError('Not supported activation type!')

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass
