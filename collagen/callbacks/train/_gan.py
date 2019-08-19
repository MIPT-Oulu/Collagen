from typing import Tuple

from torch.nn import Module

from collagen.core import Module, Callback
from collagen.core.utils import wrap_tuple, freeze_modules


class GANCallback(Callback):
    def __init__(self, ctype='gan_cb'):
        super().__init__(ctype=ctype)

    # GAN
    def on_gan_g_batch_begin(self, *args, **kwargs):
        pass

    def on_gan_g_batch_end(self, *args, **kwargs):
        pass

    def on_gan_d_batch_begin(self, *args, **kwargs):
        pass

    def on_gan_d_batch_end(self, *args, **kwargs):
        pass


class GeneratorBatchFreezer(GANCallback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_gan_g_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_gan_g_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)


class DiscriminatorBatchFreezer(GANCallback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_gan_d_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_gan_d_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)
