import torch.nn as nn
import torch.nn.functional as F
from collagen.core import Module


class AutoEncoder(Module):
    """
    AutoEncoder, consists of a simple Encoder and a simple Decoder.
    Encoder is designed to take 32x32 images and encode it to some latent space
    The decoder decodes the latent space to image of size 32x32
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, bw=32):
        """
        constructor initializes the AutoEncoder object
        Parameters
        ----------
        bw: int
            bandwidth of the network
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(bw=bw)
        self.decoder = Decoder(bw=bw)

    def forward(self, x):
        """
        forward pass for the AutoEncoder
        Parameters
        ----------
        x: Tensor
            data to be passed through the neural net

        Returns
        -------
        Tensor
            reconstructed data
        """
        return self.decoder(self.encoder(x))


def make_decoder_layer(inc, outc):
    """
    method for making complex decoder layers
    Parameters
    ----------
    inc: int
        number of input channel(s)
    outc: int
        number of output channel(s)

    Returns
    -------
    Sequential
        a sequential neural net consisting a transposed convolution layer, batch normalization and relu layer as a
        single block

    """
    block = nn.Sequential(nn.ConvTranspose2d(inc, outc, 4, 2, 1, bias=False),
                          nn.BatchNorm2d(num_features=outc),
                          nn.ReLU(True))
    return block


def make_encoder_layer(inc, outc):
    """
    makes complex encoder layers
    Parameters
    ----------
    inc: int
        number of input channel(s)
    outc: int
        number of output channel(s)

    Returns
    -------
    Sequential
        a sequential neural net consisting a convolution layer, batch normalization and relu layer as a single
        block
    """
    block = nn.Sequential(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1),
                          nn.BatchNorm2d(num_features=outc),
                          nn.ReLU(True))
    return block


class Encoder(Module):
    """
    Encoder class encompasses a separate neural network as an encoder
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, n_z=100, bw=32, n_inp=3):
        """
        constructor
        Parameters
        ----------
        n_z: int
            dimension of the latent space
        bw: int
            bandwidth of the network
        n_inp: int
            number of input channel(s)
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
        forward pass of the Encoder
        Parameters
        ----------
        x: Tensor
            data to be passed through the network

        Returns
        -------
        Tensor
            Encoded latent space
        """
        o = F.max_pool2d(self.block1(x), 2)
        o = F.max_pool2d(self.block2(o), 2)
        o = F.max_pool2d(self.block3(o), 2)
        o = F.max_pool2d(self.block4(o), 2)
        o = F.max_pool2d(self.block5(o), 2)
        return self.hidden(o)


class Decoder(Module):
    """
    Decoder class hosting functions for decoding data from latent space
    """
    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass

    def __init__(self, n_z=100, bw=32, n_out=3):
        """
        constructor
        Parameters
        ----------
        n_z: int
            dimension of the latent space
        bw: int
            bandwidth of the network
        n_out: int
            number of output channel(s)
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
        forward pass for the decoder
        Parameters
        ----------
        x: Tensor
            latent space to be decoded

        Returns
        -------
        Tensor
            decoded latent space

        """
        o = self.block1(x)
        o = self.block2(o)
        o = self.block3(o)
        o = self.block4(o)
        o = self.block5(o)
        return self.out(o)
