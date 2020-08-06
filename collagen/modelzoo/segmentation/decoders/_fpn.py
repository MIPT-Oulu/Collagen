import torch
import torch.nn.functional as F
from torch import nn

from collagen.core import Module
from collagen.modelzoo.modules import ConvBlock


class FPNBlock(Module):
    """
    Extended implementation from https://github.com/qubvel/segmentation_models.pytorch

    """

    def __init__(self, pyramid_channels, skip_channels, dropout=None):
        super(FPNBlock, self).__init__(None, None)
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.dropout = dropout
        self.dropout_on = False

    def switch_dropout(self):
        self.dropout_on = not self.dropout_on

    def forward(self, x):
        x, skip = x
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training or self.dropout_on)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x


class SegmentationBlock(Module):
    """
    Extended implementation from https://github.com/qubvel/segmentation_models.pytorch

    """

    def __init__(self, in_channels, out_channels, n_upsamples=0, normalization='BN'):
        super(SegmentationBlock, self).__init__(None, None)

        blocks = [
            ConvBlock(ks=3, inp=in_channels, out=out_channels, stride=1, pad=1,
                      activation='relu',
                      normalization=normalization, bias=False),
        ]
        if n_upsamples > 0:
            blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(ConvBlock(ks=3, inp=out_channels, out=out_channels, stride=1, pad=1,
                                        activation='relu',
                                        normalization=normalization, bias=False
                                        ))
                blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass


class FPNDecoder(Module):
    """
    Extended implementation from https://github.com/qubvel/segmentation_models.pytorch

    """

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            spatial_dropout=0.2,
            normalization='BN',
            bayesian_dropout=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1], dropout=bayesian_dropout)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2], dropout=bayesian_dropout)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3], dropout=bayesian_dropout)

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3,
                                    normalization=normalization)

        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2,
                                    normalization=normalization)

        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1,
                                    normalization=normalization)

        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0,
                                    normalization=normalization)

        self.assembly = ConvBlock(ks=3, inp=segmentation_channels * 4,
                                  out=segmentation_channels, stride=1, pad=1,
                                  activation='relu', normalization=normalization)

        self.spatial_dropout = spatial_dropout

        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

        self.initialize()

    def switch_dropout(self):
        self.p4.switch_dropout()
        self.p3.switch_dropout()
        self.p2.switch_dropout()

    def forward(self, x, out_shape=None):
        _, c2, c3, c4, c5 = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        x = self.assembly(torch.cat((s5, s4, s3, s2), 1))

        if self.spatial_dropout is not None:
            x = F.dropout2d(x, self.spatial_dropout, training=self.training)

        x = self.final_conv(x)

        if out_shape is None:
            return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            return F.interpolate(x, size=out_shape, mode='bilinear', align_corners=True)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass
