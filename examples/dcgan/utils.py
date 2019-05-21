import argparse
import torch
import numpy as np
import solt.data as sld
from collagen.data.utils import ApplyTransform, Normalize, Compose
import solt.transforms as slt
import random


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')


def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img/255.0, np.float32(target)


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


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=64, help='Latent space size')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--ngpu', type=int, default=1, help='Num of GPUs')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="dcgan", help='Comment of log')
    parser.add_argument('--grid_shape', type=int, default=8, help='Shape of grid of generated images')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return args
