import numpy as np
import pandas as pd
from torchvision import datasets
from torch import Tensor

from collagen.core.utils import to_cpu

__all__ = ["get_cifar10", "get_mnist"]


def get_mnist(data_folder='.', train=True):
    _db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    # list_rows = [{"data": to_cpu(item) if isinstance(item, Tensor) else np.array(item),
    #               "target": to_cpu(target)[0] if isinstance(target, Tensor) else target} for item, target in _db]

    list_rows = []
    for item, cls in _db:
        data = to_cpu(item) if isinstance(item, Tensor) else np.array(item)
        target = np.asscalar(cls) if isinstance(cls, Tensor) else cls
        list_rows.append({'data': data, 'target': target})

    meta_data = pd.DataFrame(list_rows)
    return meta_data, list(range(10))


def get_cifar10(data_folder='.', train=True):
    _db = datasets.CIFAR10(data_folder, train=train, transform=None, download=True)
    list_rows = [{"data": _db.data[i, :, :, :], "target": _db.targets[i]} for i in
                 range(len(_db.targets))]
    meta_data = pd.DataFrame(list_rows)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return meta_data, classes
