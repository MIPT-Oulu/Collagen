import torch
from typing import Tuple, Any, List
from collagen.core import KVS
from collagen.core import Module


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

    """
    def __init__(self, module: Module, optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module):

        self.__module: Module = module
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__loss: torch.nn.Module = loss
        self.__kvs: KVS = KVS()

        # Params of ``backward``
        self.__retain_graph: bool or None = None
        self.__create_graph: bool = False
        self.__gradient = None

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, new_loss: torch.nn.Module):
        self.__loss: torch.nn.Module = new_loss

    def optimizer_params(self, param_name):
        """Returns the value of optimizer parameter for every group of trainable parameters.
        """
        return [(group['name'], group[param_name]) for group in self.__optimizer.param_groups]

    def set_optimizer_param(self, param_name: str, new_value: Tuple[str, float] or float):
        """Sets a parameter of the optimizer for a particular group of trainable parameters or all groups.

        Parameters
        ----------
        param_name : str
            Name of the optimizer's parameters, e.g. `lr`, `weight_decay` `momentum` etc.
        new_value : Tuple[str, float] or float
            Value of the new parameter. If Tuple, then the first value int specifies the parameters group,
            and the second specifies the actual value.

        """
        for group in self.__optimizer.param_groups:
            if isinstance(new_value, float):
                group[param_name] = new_value[1]
            else:
                if new_value[0] == group['name']:
                    group[param_name] = new_value[1]

    def set_backward_param(self, gradient=None, retain_graph=None, create_graph=False):
        self.__gradient = gradient
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def add_param_group(self, group_name: str):
        """Adds parameter group to the optimizer.

        Parameters
        ----------
        group_name : str
            Name of the group, which needs to be added from model.

        """
        self.__optimizer.add_param_group(self.__module.parameters(group_name))

    def train_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                   target: torch.Tensor or Tuple[torch.Tensor],
                   accumulate_grad: bool = False, return_out=False,
                   callbacks=Tuple[callable] or List[callable] or None) -> float:
        """
        Performs one training iteration using the given mini-batch.

        Parameters
        ----------
        batch : torch.Tensor or Tuple[torch.Tensor]
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        accumulate_grad : bool
            Whether to zero grad before computing the new gradients.
            False by default, but if True, then the gradients can be accumulated.
            Useful if the batch size are too small because of the input size.
        return_out : bool
            Whether to return output
        callbacks : Tuple[callable] or List[callable] or None
            Callbacks to be used during the training.
        Returns
        -------
        out : float
            Value of the loss

        """

        if not accumulate_grad and self.__optimizer is not None:
            self.__optimizer.zero_grad()

        return self.__batch_step(batch=batch, target=target, with_grad=True,
                                 with_backward=True,
                                 return_out=return_out, callbacks=callbacks)

    def eval_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                  target: torch.Tensor or Tuple[torch.Tensor],
                  return_out=False,
                  callbacks=Tuple[callable] or List[callable] or None) -> Tuple[float, torch.Tensor or tuple] or float:
        """
        Performs evaluation of the given mini-batch. If needed, also returns the results.

        Parameters
        ----------
        batch : torch.Tensor or Tuple[torch.Tensor]
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        return_out : bool
            Whether to return the output of the network
        callbacks : Tuple[callable] or List [callable] or None
            Callbacks to be used during the training.
        Returns
        -------
        out : Tuple[float, torch.Tensor or tuple] or float
            Result of the evaluation
        """

        return self.__batch_step(batch, target, with_grad=False,
                                 with_backward=False,
                                 eval_mode=True,
                                 return_out=return_out, callbacks=callbacks)

    def __batch_step(self, batch: torch.Tensor or Tuple[torch.Tensor],
                     target: torch.Tensor or Tuple[torch.Tensor],  with_grad: bool = True,
                     with_backward: bool = True, eval_mode: bool = False,
                     return_out: bool = False,
                     callbacks=Tuple[callable] or List[callable] or None) -> Tuple[float, Any] or float:
        """
        Private method, which handles the logic for training and evaluation for 1 mini-batch.

        Parameters
        ----------
        batch : torch.Tensor
            Mini-batch
        target : torch.Tensor or Tuple[torch.Tensor]
            One or multiple targets
        with_grad : bool
            Whether to evaluate the given batch with gradient
        with_backward : bool
            Whether to perform a backward pass
        eval_mode : bool
            Whether to switch the trained module to the evaluation mode
        return_out : bool
            Whether to return the output
        callbacks : Tuple[callable] or List [callable] or None
            Callbacks to be used during the batch step.

        Returns
        -------
            out : Tuple[float, torch.Tensor or tuple] or float
                Loss value and possibly the output of the model.
        """

        module_device = next(self.__module.parameters()).device
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
            if isinstance(batch, tuple) and len(batch) == 1:
                batch = batch[0]
            if isinstance(target, tuple) and len(target) == 1:
                target = target[0]

            # Transfer input and target into proper device
            if isinstance(batch, tuple) or isinstance(batch, list):
                batch_on_device = tuple([b.to(module_device) for b in batch])
            else:
                batch_on_device = batch.to(module_device)

            if isinstance(target, tuple) or isinstance(target, list):
                target_on_device = tuple([t.to(module_device) for t in target])
            else:
                target_on_device = target.to(module_device)

            # Forward
            for cb in callbacks:
                cb.on_forward_begin(module=self.__module,
                                    input=batch_on_device,
                                    target=target_on_device,
                                    optimizer=self.__optimizer,
                                    criterion=self.__loss)

            out = self.__module(batch_on_device)

            for cb in callbacks:
                cb.on_forward_end(module=self.__module,
                                  input=batch_on_device,
                                  target=target_on_device,
                                  output=out,
                                  optimizer=self.__optimizer,
                                  criterion=self.__loss)

            # Compute loss
            for cb in callbacks:
                cb.on_loss_begin(session=self,
                                 input=batch_on_device,
                                 target=target_on_device,
                                 output=out)

            loss = self.__loss(out, target_on_device)

            for cb in callbacks:
                cb.on_loss_end(session=self,
                               loss=loss,
                               input=batch_on_device,
                               target=target_on_device,
                               output=out)

            if with_backward:
                # Backward
                for cb in callbacks:
                    cb.on_backward_begin(session=self,
                                         loss=loss,
                                         input=batch_on_device,
                                         target=target_on_device,
                                         output=out)

                loss.backward(gradient=self.__gradient,
                              retain_graph=self.__retain_graph,
                              create_graph=self.__create_graph)

                for cb in callbacks:
                    cb.on_backward_end(session=self,
                                       loss=loss,
                                       input=batch_on_device,
                                       target=target_on_device,
                                       output=out,
                                       optimizer=self.__optimizer,
                                       criterion=self.__loss)

                # Optimizer step
                for cb in callbacks:
                    cb.on_optimizer_step_begin(module=self.__module,
                                               loss=loss,
                                               input=batch_on_device,
                                               target=target_on_device,
                                               output=out,
                                               optimizer=self.__optimizer,
                                               criterion=self.__loss)

                self.__optimizer.step()

                for cb in callbacks:
                    cb.on_optimizer_step_begin(module=self.__module,
                                               loss=loss,
                                               input=batch_on_device,
                                               target=target_on_device,
                                               output=out,
                                               optimizer=self.__optimizer,
                                               criterion=self.__loss)

            if not return_out:
                return loss.item()
            else:
                return loss.item(), out










