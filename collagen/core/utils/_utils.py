from typing import Tuple
import torch

__all__ = ["to_cpu", "wrap_tuple", "auto_detect_device", "freeze_modules"]


def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):
    x_cpu = x

    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            if use_numpy:
                x_cpu = x.to('cpu').detach().numpy()
            elif required_grad:
                x_cpu = x.to('cpu')
            else:
                x_cpu = x.to('cpu').required_grad_(False)
        elif use_numpy:
            if x.requires_grad:
                x_cpu = x.detach().numpy()
            else:
                x_cpu = x.numpy()

    return x_cpu


def wrap_tuple(x):
    if not isinstance(x, tuple) and not isinstance(x, list):
        return (x,)
    else:
        return x


def auto_detect_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_modules(modules: torch.nn.Module or Tuple[torch.nn.Module], invert=False):
    requires_grad = invert
    _modules = wrap_tuple(modules)
    for md in _modules:
        # md.train(requires_grad)
        for param in md.parameters():
            param.requires_grad = requires_grad
