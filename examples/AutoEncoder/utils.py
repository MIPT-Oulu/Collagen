# Some functions are copied and/or modified from cnn examples
import argparse
import torch
import solt.data as sld
import solt.core as slc
import solt.transforms as slt
from torchvision import datasets
import pandas as pd
import numpy as np
from collagen.data.utils import ApplyTransform, Normalize, Compose


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bw', type=int, default=64, help='Bandwidth of model')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--snapshots', default='snapshots', help='Where to save the snapshots')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset name')
    parser.add_argument('--device', type=str, default="cuda:0", help='Use `cuda` or `cpu`')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    args = parser.parse_args()

    return args


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img, target


def init_mnist_transforms(n_channels=1):
    if n_channels == 1:
        norm_mean_std = Normalize((0.5,), (1.0,))
    elif n_channels == 3:
        norm_mean_std = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=34),
            slt.CropTransform(crop_size=32, crop_mode='r')
        ]),
        unpack_solt,
         ApplyTransform(norm_mean_std)
    ])

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
         ApplyTransform(norm_mean_std)
    ])

    return train_trf, test_trf


def get_mnist32x32(data_folder='.', train=True):
    """
    get_mnist32x32 will return the mnist data as 32x32 instead of 28x28
    :param data_folder: string, where the data will be stored
    :param train: boolean, if the data is training or testing data
    :return: panda DataFrame and the classes
    """
    _db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    list_rows = [{"data": to32x32(item), "target": cast2int(target)} for item, target in _db]
    meta_data = pd.DataFrame(list_rows)
    return meta_data, list(range(10))


def get_mnist32x32byClass(data_folder='.', train=True, class_id=0):
    """
    get_mnist32x32byClass is an extension of get_mnist32x32, instead of returning all the
    images, this method returns the images of the class specified
    :param data_folder: str, determines where to save the data
    :param train: boolean, if the data is for training or testing
    :param class_id: int, from 0 to 9 for MNIST, if the class_id is 0, the method will
    return all the images of 0
    :return: panda DataFrame and class_id as list
    """
    _db = datasets.MNIST(data_folder, train=train, transform=None, download=True)
    list_rows= []
    for item, target in _db:
        if cast2int(target) == class_id:
            list_rows.append({"data": to32x32(item), "target": cast2int(target)})
    meta_data = pd.DataFrame(list_rows)
    return meta_data, [class_id]


def to32x32(data):
    """
    to32x32 zero pads the 28x28 data to 32x32
    :param data: array of size 28x28
    :return: array of size 32x32
    """
    tmp = np.zeros((32, 32), dtype=np.float32)
    data = np.asarray(data, dtype=np.float32)
    if np.max(data) > 1.0:
        tmp[2:30, 2:30] = data/255.0
    else:
        tmp[2:30, 2:30] = data
    return tmp


def cast2int(data):
    """
    cast2int type casts data to integer
    :param data: any numeric value
    :return: integer of tensor or other type
    """
    if isinstance(data, int):
        return data
    else:
        return data.item()