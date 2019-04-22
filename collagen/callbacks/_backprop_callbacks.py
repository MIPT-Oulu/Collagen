from typing import Tuple
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from collagen.core import Callback, Module
from collagen.core.utils import to_tuple


class BackwardCallback(Callback):
    def __init__(self, retain_graph=True, create_graph=False):
        super().__init__()
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def on_backward_begin(self, session, **kwargs):
        session.set_backward_param(retain_graph=self.__retain_graph, create_graph=self.__create_graph)


class ClipGradCallback(Callback):
    modes = ["norm", "value"]

    def __init__(self, modules: Tuple[Module] or Module, mode: str = "norm", **kwargs):
        super().__init__(ctype="tuner")
        if mode not in self.modes:
            raise ValueError("Mode must be in {}".format(self.modes))

        self.__mode: str = mode
        self.__modules: Tuple[Module] = to_tuple(modules)
        self.__kwargs = kwargs

    def on_backward_end(self, *args, **kwargs):
        if self.__mode == "value":
            for md in self.__modules:
                clip_grad_value_(md.parameters(), **self.__kwargs)
        else:
            for md in self.__modules:
                clip_grad_norm_(md.parameters(), **self.__kwargs)
