import argparse
import random

import numpy as np
import solt.data as sld
import solt.transforms as slt
import torch

from collagen.data.utils import ApplyTransform, Normalize, Compose


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img / 255.0, np.float32(target)


def init_mnist_transforms():
    return Compose([
        wrap2solt,
        slt.ResizeTransform(resize_to=(64, 64), interpolation='bilinear'),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])


def parse_item_mnist_gan(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {data_key: img, target_key: np.float32(1.0), 'class': target}



