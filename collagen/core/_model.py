import torch


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.__param_groups = dict()

    def parameters(self, group_name=None, name=None):
        if group_name is None:
            return super(Module, self).parameters()
        else:
            if name is None:
                return self.__param_groups[group_name]
            else:
                return self.__param_groups[group_name][name]

    def _add_to(self, layer, name, group_name):
        if group_name not in self.__param_groups:
            self.__param_groups[group_name] = {}

        self.__param_groups[group_name][name] = layer.parameters()

    def forward(self, *input):
        raise NotImplementedError
