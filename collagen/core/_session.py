import torch
from torch.optim import Optimizer

from ..logging import KVS
from ._model import CModule
from ..data import DataProvider


class Session(object):
    """Session class, which implements the basic logic of the training loop.

    Parameters
    ----------
    module : CModule
        Instantiated collagen module with trainable parameters.

    """
    def __init__(self, module: CModule, optimizer: Optimizer,
                 loss: torch.nn.Module, data_provider: DataProvider,
                 loader_id: str):

        self.__module = module
        self.__optimizer = optimizer
        self.__data_provider = data_provider
        self.__loss = loss
        self.__loader_id = loader_id
        self.__kvs = KVS()

        self.__optimizer.add_param_group(self.__module.parameters())

    def train_step(self, batch, with_backward=True, zero_grad=True):
        if zero_grad:
            self.__optimizer.zero_grad()
        return self.batch_step(batch, with_grad=True,
                               with_backward=with_backward)

    def eval_step(self, batch, return_out=False):
        return self.batch_step(batch, with_grad=False,
                               with_backward=False,
                               eval_mode=True,
                               return_out=return_out)

    def batch_step(self, batch, with_grad=True, with_backward=True, eval_mode=False, return_out=False):
        if eval_mode:
            with_backward = False
            with_grad = False
            self.__module.train(False)
        else:
            self.__module.train(True)

        if with_backward:
            if not with_grad:
                raise ValueError

        with torch.set_grad_enabled(with_grad):
            out = self.__module(batch)
            loss = self.__loss(out)

        if with_backward:
            loss.backward()
            self.__optimizer.step()
        if not return_out:
            return loss
        else:
            return loss, return_out










