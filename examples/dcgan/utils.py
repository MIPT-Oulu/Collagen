import numpy as np
import cv2
from solt import Stream
import solt.transforms as slt


def init_mnist_transforms():
    return Stream([slt.Pad(pad_to=(32, 32))])


def parse_item_mnist_gan(root, entry, trf, data_key, target_key):

    img = entry[data_key]
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    if img.shape[-1] == 1:
        stats = {'mean': (0.5,), 'std': (0.5,)}
    elif img.shape[-1] == 3:
        stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    else:
        raise ValueError("Not support channels of {}".format(img.shape[-1]))

    trf_data = trf({'image': img}, normalize=True, **stats)
    return {data_key: trf_data['image'], target_key: np.float32(1.0), 'class': entry[target_key]}
