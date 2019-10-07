import torch.nn.functional as F
from torch import nn
import torch
from collagen.core import Module
from collagen.modelzoo.modules import ConvBlock


class DecoderBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, depth=2, mode='nearest', activation='relu', normalization='BN'):
        super(DecoderBlock, self).__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        self.layers = nn.Sequential()

        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock(3, inp=inp_channels,
                                     out=out_channels, activation=activation, normalization=normalization, bias=False))
            else:
                tmp.append(ConvBlock(3, inp=out_channels,
                                     out=out_channels, activation=activation, normalization=normalization, bias=False))
            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        if self.ups_mode == 'bilinear':
            o = F.interpolate(x, scale_factor=2, mode=self.ups_mode, align_corners=True)
        else:
            o = F.interpolate(x, scale_factor=2, mode=self.ups_mode)
        o = self.layers(o)
        return o


class UNetDecoder(Module):
    def __init__(self, encoder_channels,
                 width=32,
                 final_channels=1,
                 activation='relu',
                 spatial_dropout=0.2,
                 normalization='BN',
                 upsample='bilinear'):
        super(UNetDecoder, self).__init__()
        self.center = DecoderBlock(encoder_channels[0], width * 8)
        self.d5 = DecoderBlock(encoder_channels[0] + width * 8, width * 8)
        self.d4 = DecoderBlock(encoder_channels[1] + width * 8, width * 8)
        self.d3 = DecoderBlock(encoder_channels[2] + width * 8, width * 2)
        self.d2 = DecoderBlock(encoder_channels[3] + width * 2, width * 2)
        self.d1 = DecoderBlock(encoder_channels[4] + width * 2, width)

        self.final_conv = nn.Conv2d(width, final_channels, kernel_size=1, padding=0)
        self.spatial_dropout = spatial_dropout

        self.initialize()

    def forward(self, x):
        c1, c2, c3, c4, c5 = x

        p6 = self.center(F.max_pool2d(c5, 2))
        p5 = self.d5(torch.cat([p6, c5], 1))
        p4 = self.d4(torch.cat([p5, c4], 1))
        p3 = self.d3(torch.cat([p4, c3], 1))
        p2 = self.d2(torch.cat([p3, c2], 1))
        p1 = self.d1(torch.cat([p2, c1], 1))

        if self.spatial_dropout is not None:
            p1 = F.dropout2d(p1, self.spatial_dropout, training=self.training)

        return self.final_conv(p1)



