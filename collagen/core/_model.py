import torch


class CModule(object):
    def __init__(self, module: torch.nn.Module, param_getter, param_init):
        self.__module = module
        self.__param_getter = param_getter
        self.__param_init = param_init

    def parameters(self):
        self.__param_getter(self.__module)

    @property
    def param_getter(self):
        return self.__param_getter

    @param_getter.setter
    def param_getter(self, new_param_getter):
        self.__param_getter = new_param_getter

    @property
    def param_init_cb(self):
        return self.__param_init

    @param_init_cb.setter
    def param_init_cb(self, new_param_init_cb):
        self.__param_init = new_param_init_cb

    def train(self, state):
        self.__module.train(state)

    def __call__(self, *args, **kwargs):
        return self.__module(args, kwargs)

