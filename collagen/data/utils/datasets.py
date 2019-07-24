import numpy as np
import pandas as pd
from torchvision import datasets


def get_mnist(data_folder='.', train=True):
    _db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    list_rows = [{"data": np.array(item), "target": target} for item, target in _db]
    meta_data = pd.DataFrame(list_rows)
    return meta_data, list(range(10))


def get_cifar10(data_folder='.', train=True):
    _db = datasets.CIFAR10(data_folder, train=train, transform=None, download=True)
    list_rows = [{"data": _db.data[i, :, :, :], "target": _db.targets[i]} for i in
                 range(len(_db.targets))]
    meta_data = pd.DataFrame(list_rows)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return meta_data, classes
