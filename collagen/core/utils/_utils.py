import torch
from typing import Tuple


def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.cpu().data.numpy()
            else:
                x_cpu = x.cpu().data
        elif use_numpy:
            x_cpu = x.numpy()

    return x_cpu


def to_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    else:
        return x


def auto_detect_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_modules(modules: torch.nn.Module or Tuple[torch.nn.Module], invert=False):
    requires_grad = invert
    _modules = to_tuple(modules)
    for md in _modules:
        # md.train(requires_grad)
        for param in md.parameters():
            param.requires_grad = requires_grad
