import torch

from ..logging import KVS
from ._model import Module


from typing import Tuple, Any


class Session(object):
    """Session class, which implements the basic logic of the training loop.

    Current implementation allows to easily set-up gradient accumulation
    and other strategies.

    Parameters
    ----------
    module : Module
        Instantiated collagen module with trainable parameters.
    optimizer : torch.Optimizer
        Optimizer to train teh model
    loss : torch.nn.Module
        Loss used in the session
    param_groups : str or tuple
        Groups of parameters, which need to be optimized. If str, then a particular group of parameters will be used.
        If a tuple of strings, then all the mentioned groups will be used.

    """
    def __init__(self, module: Module, optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module, param_groups: str or Tuple[str]):

        self.__module: Module = module
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__loss: torch.nn.Module = loss
        self.__kvs: KVS = KVS()
        self.__param_groups: str or Tuple[str] = param_groups

        if isinstance(param_groups, tuple):
            for group_name in param_groups:
                self.add_param_group(group_name)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, new_loss: torch.nn.Module):
        self.__loss: torch.nn.Module = new_loss

    def add_param_group(self, group_name: str):
        self.__optimizer.add_param_group(self.__module.parameters(group_name))

    def train_step(self, batch: torch.Tensor, with_backward: bool = True,
                   zero_grad: bool = True) -> Tuple[float, Any] or float:

        if zero_grad:
            self.__optimizer.zero_grad()

        return self.batch_step(batch, with_grad=True, with_backward=with_backward)

    def eval_step(self, batch: torch.Tensor, return_out=False) -> Tuple[float, torch.Tensor or tuple] or float:

        return self.batch_step(batch, with_grad=False,
                               with_backward=False,
                               eval_mode=True,
                               return_out=return_out)

    def batch_step(self, batch: torch.Tensor, with_grad: bool = True,
                   with_backward: bool = True, eval_mode: bool = False,
                   return_out: bool = False) -> Tuple[float, Any] or float:

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
            return loss.item()
        else:
            return loss.item(), return_out










