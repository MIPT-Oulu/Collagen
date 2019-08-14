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


class UpdateEMA(Callback):
    def __init__(self, st_model, te_model, decay=0.97):
        super().__init__(ctype="update_weight")
        self.__st_model = st_model
        self.__te_model = te_model
        self.__alpha = decay

    def on_batch_end(self, epoch, *args, **kwargs):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (epoch + 1), self.__alpha)
        for ema_param, param in zip(self.__te_model.parameters(), self.__st_model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class SetTeacherTrain(Callback):
    def __init__(self, te_model):
        super().__init__(ctype='custom')
        self.__te_model = te_model

    def on_batch_begin(self, *args, **kwargs):
        self.__te_model.train(True)
