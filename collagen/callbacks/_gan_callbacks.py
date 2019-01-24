from typing import Tuple
import torch
from collagen.core import Module, Callback
from collagen.data.utils import to_tuple, freeze_modules


class OnGeneratorBatchFreezer(Callback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(type="freezer")
        self.__modules: Tuple[Module] = to_tuple(modules)

    def on_gan_g_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_gan_g_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)


class OnDiscriminatorBatchFreezer(Callback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(type="freezer")
        self.__modules: Tuple[Module] = to_tuple(modules)

    def on_gan_d_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_gan_d_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)


class GeneratorLoss(torch.nn.Module):
    def __init__(self, d_network, d_loss):
        super(GeneratorLoss, self).__init__()
        self.__d_network = d_network
        self.__d_loss = d_loss

    def forward(self, img: torch.Tensor, target: torch.Tensor):
        output = self.__d_network(img)
        loss = self.__d_loss(output, 1 - target)
        return loss
