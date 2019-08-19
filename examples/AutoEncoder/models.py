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

    def __init__(self):
        """
        @constructor
        """
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        """
        @method: forward
        :param x: Tensor, typically of the size Batch * Channel * 32 * 32
        :return: Tensor of the same size of x which is a reconstruction of the input
        """
        return self.decoder(self.encoder(x))


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

    def __init__(self, n_z=100):
        """
        @constructor, initializes the Encoder object
        :param n_z: Integer, the dimension of latent space
        """
        super(Encoder, self).__init__()
        self.n_z = n_z
        layers = []
        # input size would be BxCx32x32
        layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1))
        # output size would be Bx8x28x28
        layers.append(nn.BatchNorm2d(num_features=8))
        layers.append(nn.LeakyReLU(negative_slope=0.2))

        inc = 8
        outc = inc * 2
        for _ in range(5):
            layers.append(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(num_features=outc))
            layers.append(nn.MaxPool2d(kernel_size=2))
            layers.append(nn.LeakyReLU(negative_slope=0.2))

            inc = outc
            outc = inc * 2
        # output size would be B*256x1x1
        layers.append(FlatLayer())
        # output size would be Bx256
        layers.append(nn.Linear(in_features=256, out_features=self.n_z))
        layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        @method: forward, does the job of forward pass
        :param x: Tensor, typically of the size batch * channel * 32 * 32
        :return: latent representation of the learned image, typically of the size batch*n_z
        """
        return self.encoder(x)


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

    def __init__(self, n_z=100):
        """
        @constructor: initializes the decoder object
        :param n_z: integer, dimension of the latent space
        """
        super(Decoder, self).__init__()
        self.n_z = n_z
        layers = []
        # input size Bx100
        layers.append(nn.Linear(in_features=self.n_z, out_features=256))
        layers.append(nn.Tanh())
        # output size Bx256
        layers.append(DeFlatLayer())
        # output size Bx256x1x1
        inc = 256
        outc = int(inc / 2)
        for _ in range(5):
            layers.append(nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(num_features=outc))
            layers.append(Upsample())
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            inc = outc
            outc = int(inc/2)
        # output size Bx8x32x32
        # DeConvolution
        layers.append(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        @method forward, does the forward pass for decoder network
        :param x: Tensor, typically of size batch * n_z
        :return: Tensor, typically of size batch * channel * 32 * 32
        """
        return self.decoder(x)


class DeFlatLayer(Module):
    """
    @class DeFlatLayer
    makes the reshape function a layer
    """
    def __init__(self):
        super(DeFlatLayer, self).__init__()

    def forward(self, x):
        """
        @method forward, does the forward pass for DeFlatLayer
        :param x: Tensor, typically of size batch * n_z
        :return: Tensor, typically of size batch * channel * 1 * 1
        """
        b, _ = x.size()
        return x.reshape(b, -1, 1, 1)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass


class FlatLayer(Module):
    """
    @class FlatLayer
    makes the reshape function a layer
    """
    def __init__(self):
        super(FlatLayer, self).__init__()

    def forward(self, x):
        """
        @method forward, does the forward pass for FlatLayer
        :param x: Tensor, typically of size batch * channel * 1 * 1
        :return: Tensor, typically of size batch * channel
        """
        b, _, _, _ = x.size()
        return x.reshape(b, -1)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass


class Upsample(Module):
    """
    @class Upsample: converts the interpolate function to a layer
    """
    def __init__(self, scale_factor=2, mode='bilinear'):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.upsample = F.interpolate

    def forward(self, x):
        return self.upsample(input=x, scale_factor=self.scale, mode=self.mode)

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass
