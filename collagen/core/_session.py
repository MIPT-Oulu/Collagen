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

    def train_step(self, batch: torch.Tensor, accumulate_grad: bool = False) -> float:
        """
        Performs one training iteration using the given mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            Mini-batch
        accumulate_grad : bool
            Whether to zero grad before computing the new gradients.
            False by default, but if True, then the gradients can be accumulated.
            Useful if the batch size are too small because of the input size.

        Returns
        -------
        out : float
            Value of the loss

        """

        if not accumulate_grad:
            self.__optimizer.zero_grad()

        return self.__batch_step(batch, with_grad=True, with_backward=True, return_out=False)

    def eval_step(self, batch: torch.Tensor, return_out=False) -> Tuple[float, torch.Tensor or tuple] or float:
        """
        Performs evaluation of the given mini-batch. If needed, also returns the results.

        Parameters
        ----------
        batch : torch.Tensor
            Mini-batch
        return_out : bool
            Whether to return the output of the network

        Returns
        -------
        out : Tuple[float, torch.Tensor or tuple] or float
            Result of the evaluation
        """

        return self.__batch_step(batch, with_grad=False,
                                 with_backward=False,
                                 eval_mode=True,
                                 return_out=return_out)

    def __batch_step(self, batch: torch.Tensor, with_grad: bool = True,
                     with_backward: bool = True, eval_mode: bool = False,
                     return_out: bool = False) -> Tuple[float, Any] or float:
        """
        Private method, which handles the logic for training and evaluation for 1 mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            Mini-batch
        with_grad : bool
            Whether to evaluate the given batch with gradient
        with_backward : bool
            Whether to perform a backward pass
        eval_mode : bool
            Whether to switch the trained module to the evaluation mode
        return_out : bool
            Whether to return the output

        Returns
        -------
            out : Tuple[float, torch.Tensor or tuple] or float
                Loss value and possibly the output of the model.
        """

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










