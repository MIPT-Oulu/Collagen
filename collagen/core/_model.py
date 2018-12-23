import torch
from typing import Tuple
from abc import abstractmethod


class Module(torch.nn.Module):
    """
    Generic building block, which assumes to have trainable parameters within it.

    This extension allows to group the layers and have an easy access to them via group names.

    """
    def __init__(self):
        super(Module, self).__init__()
        self.__param_groups = dict()

    def parameters(self, group_names: str or Tuple[str] or None = None,
                   name: str or None = None):
        """
        Returns an iterator through the parameters of the module from one or many groups.

        Also allows to retrieve a particular module from a group using its name.

        Parameters
        ----------
        group_names: str or Tuple[str] or None
            Parameter group names.
        name: str or None
            Name of the layer from the group to be returned. Should be set to None
            if all the parameters from the group are needed.

        Yields
        -------
        Parameter: torch.nn.Parameter
            Module parameter

        """
        if group_names is None:
            return super(Module, self).parameters()
        else:
            if name is None:
                if isinstance(group_names, str):
                    group_names = (group_names, str)
                for group_name in group_names:
                    yield self.__param_groups[group_name]
            else:
                if not isinstance(group_names, str):
                    raise ValueError
                yield {'params': self.__param_groups[group_names][name], 'name': name}

    def add_to(self, layer: torch.nn.Module, name: str, group_names: str or Tuple[str]):
        """
        Adds a layer with trainable parameters to one or several groups.

        Parameters
        ----------
        layer : torch.nn.Module
            The layer to be added to the group(s)
        name : str
            Name of the layer
        group_names: str Tuple[str]
            Group names.

        """
        if name is None or group_names is None:
            raise ValueError
        for group_name in group_names:
            if group_name not in self.__param_groups:
                self.__param_groups[group_name] = {}

            self.__param_groups[group_name][name] = layer.parameters()

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError
