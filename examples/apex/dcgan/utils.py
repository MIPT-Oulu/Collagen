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


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="dcgan", help='Comment of log')
    parser.add_argument('--grid_shape', type=int, default=8, help='Shape of grid of generated images')
    parser.add_argument('--mms', type=bool, default=True, help='Shape of grid of generated images')
    parser.add_argument('--distributed', type=bool, default=False, help='whether to use DDP')
    parser.add_argument('--gpu', type=int, default=0, help='Default GPU id')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--use_apex', default=False, type=bool, help='Whether to use apex library')
    parser.add_argument('--loss_scale', default=None, type=int, help='loss scale for apex amp')
    parser.add_argument('--n_channels', default=None, type=int, help='number of input channels')
    parser.add_argument('--opt_level', default='O1', type=str, help='number of input channels')
    parser.add_argument('--suppress_warning', default=True, type=bool, help='whether to print warning messages')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers ')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return args
