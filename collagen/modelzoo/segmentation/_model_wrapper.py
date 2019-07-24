from collagen.core import Module
from collagen.modelzoo.segmentation import constants
from collagen.modelzoo.segmentation import backbones
from collagen.modelzoo.segmentation import decoders


class EncoderDecoder(Module):
    def __init__(self, n_outputs, backbone: str or Module, decoder: str or Module,
                 decoder_normalization='BN', spatial_dropout=None, bayesian_dropout=None):
        super(EncoderDecoder, self).__init__()
        if isinstance(backbone, str):
            if backbone in constants.allowed_encoders:
                if 'resnet' in backbone:
                    backbone = backbones.ResNetBackbone(backbone, dropout=bayesian_dropout)
                else:
                    ValueError('Cannot find the implementation of the backbone!')
            else:
                raise ValueError('This backbone name is not in the list of allowed backbones!')

        self.backbone = backbone
        if isinstance(decoder, str):
            if decoder in constants.allowed_decoders:
                if decoder == 'FPN':
                    decoder = decoders.FPNDecoder(encoder_channels=backbone.output_shapes,
                                                  pyramid_channels=256, segmentation_channels=128,
                                                  final_channels=n_outputs, spatial_dropout=spatial_dropout,
                                                  normalization=decoder_normalization,
                                                  bayesian_dropout=bayesian_dropout)

        self.decoder = decoder
        self.decoder.initialize()

    def forward(self, x):
        features = self.backbone(x)
        return self.decoder(features)

    def switch_dropout(self):
        """
        Has effect only if the model supports monte-carlo dropout inference.

        """
        self.backbone.switch_dropout()
        self.decoder.switch_dropout()

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

