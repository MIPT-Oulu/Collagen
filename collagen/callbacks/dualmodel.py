from typing import Tuple
from torch.nn import Module
from collagen.core import Module, Callback
from collagen.core.utils import wrap_tuple, freeze_modules


class DualModelCallback(Callback):
    def __init__(self):
        super().__init__(ctype='dualmodel_cb')

    def on_m1_batch_begin(self, *args, **kwargs):
        pass

    def on_m1_batch_end(self, *args, **kwargs):
        pass

    def on_m2_batch_begin(self, *args, **kwargs):
        pass

    def on_m2_batch_end(self, *args, **kwargs):
        pass


class M1BatchFreezer(DualModelCallback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_m1_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_m1_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)


class M2BatchFreezer(DualModelCallback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_m2_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_m2_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)
