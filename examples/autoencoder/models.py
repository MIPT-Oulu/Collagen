import torch.nn as nn
import torch.nn.functional as F
from collagen.core import Module


class AutoEncoder(Module):
    """
    @class: AutoEncoder, consists of a simple Encoder and a simple Decoder.
    Encoder is designed to take 32x32 images and encode it to some latent space
    The decoder decodes the latent space to image of size 32x32
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, bw=32):
        """
        @constructor
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(bw=bw)
        self.decoder = Decoder(bw=bw)

    def forward(self, x):
        """
        @method: forward
        :param x: Tensor, typically of the size Batch * Channel * 32 * 32
        :return: Tensor of the same size of x which is a reconstruction of the input
        """
        return self.decoder(self.encoder(x))


def make_decoder_layer(inc, outc):
    block = nn.Sequential(nn.ConvTranspose2d(inc, outc, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(num_features=outc),
                          nn.ReLU(True))
    return block


def make_encoder_layer(inc, outc):
    block = nn.Sequential(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1),
                          nn.BatchNorm2d(num_features=outc),
                          nn.ReLU(True))
    return block


class Encoder(Module):
    """
    @class: Encoder, Encodes the image into latent space
    :var n_z: integer, the dimension of the latent space
    :var encoder: neural network, hosts a sequential neural network
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, n_z=100, bw=32, n_inp=3):
        """
        @constructor, initializes the Encoder object
        :param n_z: Integer, the dimension of latent space
        """
        super(Encoder, self).__init__()
        self.n_z = n_z
        self.block1 = make_encoder_layer(n_inp, bw)  # 16x16
        self.block2 = make_encoder_layer(bw, bw * 2)  # 8x8
        self.block3 = make_encoder_layer(bw * 2, bw * 4)  # 4x4
        self.block4 = make_encoder_layer(bw * 4, bw * 8)  # 2 x 2
        self.block5 = make_encoder_layer(bw * 8, bw * 8)  # 1 x 1
        self.hidden = make_encoder_layer(bw * 8, self.n_z)  # 1 x 1

    def forward(self, x):
        """
        @method: forward, does the job of forward pass
        :param x: Tensor, typically of the size batch * channel * 32 * 32
        :return: latent representation of the learned image, typically of the size batch*n_z
        """
        o = F.max_pool2d(self.block1(x), 2)
        o = F.max_pool2d(self.block2(o), 2)
        o = F.max_pool2d(self.block3(o), 2)
        o = F.max_pool2d(self.block4(o), 2)
        o = F.max_pool2d(self.block5(o), 2)
        return self.hidden(o)


class Decoder(Module):
    """
    @class: Decoder, hosts the function for decoding the learned representation
    :var n_z: integer, dimension of the latent space
    :var decoder: neural network, hosts sequential network for decoding
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, n_z=100, bw=32, n_out=3):
        """
        @constructor: initializes the decoder object
        :param n_z: integer, dimension of the latent space
        """
        super(Decoder, self).__init__()
        self.n_z = n_z
        self.n_z = n_z
        self.block1 = make_decoder_layer(self.n_z, bw * 8)  # 2x2
        self.block2 = make_decoder_layer(bw * 8, bw * 8)  # 4x4
        self.block3 = make_decoder_layer(bw * 8, bw * 4)  # 8x8
        self.block4 = make_decoder_layer(bw * 4, bw * 2)  # 16x16
        self.block5 = make_decoder_layer(bw * 2, bw)  # 32x32
        self.out = nn.Sequential(nn.Conv2d(bw, n_out, kernel_size=1, stride=1),
                                 nn.Tanh())

    def forward(self, x):
        """
        @method forward, does the forward pass for decoder network
        :param x: Tensor, typically of size batch x Z x 1 x 1
        :return: Tensor, typically of size batch * channel * 32 * 32
        """
        o = self.block1(x)
        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.block5(o)
        return self.out(o)
