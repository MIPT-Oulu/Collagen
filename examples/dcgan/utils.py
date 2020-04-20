import numpy as np
from solt import Stream
import solt.transforms as slt


def init_mnist_transforms():
    return Stream([slt.Resize(resize_to=(32, 32), interpolation='bilinear')])


def parse_item_mnist_gan(root, entry, trf, data_key, target_key):

    img = entry[data_key]
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    if img.shape[-1] == 1:
        stats = {'mean': (0.1307,), 'std': (0.3081,)}
    elif img.shape[-1] == 3:
        stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.261)}
    else:
        raise ValueError("Not support channels of {}".format(img.shape[-1]))

    trf_data = trf({'image': entry[data_key]}, normalize=True, **stats)
    return {data_key: trf_data['image'], target_key: np.float32(1.0), 'class': entry[target_key]}
