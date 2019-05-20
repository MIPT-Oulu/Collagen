from torchvision import datasets
import numpy as np
import pandas as pd


def get_mnist(data_folder='.', train=True):
    mnist_db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    list_rows = [{"data": np.array(item), "target": target.item()} for item, target in mnist_db]
    meta_data = pd.DataFrame(list_rows)
    return meta_data, list(range(10))
