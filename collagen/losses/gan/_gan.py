from torch import Tensor
from collagen.core import Module


class GeneratorLoss(Module):
    def __init__(self, d_network, d_loss):
        super(GeneratorLoss, self).__init__()
        self.__d_network = d_network
        self.__d_loss = d_loss

    def forward(self, img: Tensor, target: Tensor):
        output = self.__d_network(img)
        loss = self.__d_loss(output, 1 - target)
        return loss
