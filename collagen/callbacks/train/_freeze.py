from typing import Tuple

from collagen.core import Callback, Module
from collagen.core.utils import wrap_tuple, freeze_modules

__all__ = ["SamplingFreezer", "BatchProcFreezer"]


class SamplingFreezer(Callback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_sample_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_sample_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)


class BatchProcFreezer(Callback):
    def __init__(self, modules: Module or Tuple[Module]):
        super().__init__(ctype="freezer")
        self.__modules: Tuple[Module] = wrap_tuple(modules)

    def on_batch_begin(self, *args, **kwargs):
        freeze_modules(self.__modules)

    def on_batch_end(self, *args, **kwargs):
        freeze_modules(self.__modules, invert=True)
