import torch
from typing import Tuple, Dict
from abc import abstractmethod


class Module(torch.nn.Module):
    """
    Generic building block, which assumes to have trainable parameters within it.

    This extension allows to group the layers and have an easy access to them via group names.

    """
    def __init__(self):
        super(Module, self).__init__()
        self.__param_groups = dict()
        self.optimize_cb = None

    # Should not overwrite parameters() because it eliminates some useful params of nn.Module such as device, requires_grad, etc.
    def _parameters(self, group: str or Tuple[str] or None = None,
                    name: str or None = None) -> Dict[str, torch.nn.Parameter or str]:
        """
        Returns an iterator through the parameters of the module from one or many groups.

        Also allows to retrieve a particular module from a group using its name.

        Parameters
        ----------
        group: str or Tuple[str] or None
            Parameter group names.
        name: str or Tuple[str] or None
            Name of the module from the group to be returned. Should be set to None
            if all the parameters from the group are needed. Alternatively, multiple modules
            from the group can be returned if it is a Tuple[str].

        Yields
        -------
        Parameters: Dict[str, torch.nn.Parameter or str]
            Dictionary of parameters. Allows to get all the parameters of submodules from multiple groups,
            or particular submodules' parameters from the given group. The returned dict has always three keys:
            params (used by optimizer), name (module name) and group name (name of the parameter groups). If name is not
            specified, it will be None.

        """
        if group is None:
            yield {'params': super(Module, self).parameters(), 'name': None, 'group_name': None}
        else:
            if name is None:
                if isinstance(group, str):
                    group = (group, )
                for group_name in group:
                    yield {'params': self.__param_groups[group_name],
                           'name': None,
                           'group_name': group_name}
            else:
                if not isinstance(group, str):
                    raise ValueError

                if isinstance(name, str):
                    name = (name, )

                for module_name in name:
                    yield {'params': self.__param_groups[group][module_name], 'name': module_name, 'group_name': group}

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
    def forward(self, x):
        raise NotImplementedError
