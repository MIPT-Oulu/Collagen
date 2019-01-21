import argparse
from collagen.core import Module
from torch import nn
import torch
import numpy as np
import solt.data as sld
from collagen.data.utils import ApplyTransform, Normalize, Compose
import solt.core as slc
import solt.transforms as slt


def wrap2solt(inp):
    img, label = inp
    return sld.DataContainer((img, label), 'IL')

def unpack_solt(dc: sld.DataContainer):
    img, target = dc.data
    img, target = torch.from_numpy(img).permute(2, 0, 1).float(), target
    return img, np.float32(target)

# def unpack_solt(dc: sld.DataContainer):
#     img, target = dc.data
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     valid = torch.ones([img.shape[0], 1], dtype=torch.float32)
#     target = torch.zeros([img.shape[0], 10], dtype=torch.float32)
#     ext_target = torch.cat((valid, target), dim=-1)
#     return img, ext_target


def init_mnist_transforms():
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
        ApplyTransform(Normalize((0.5,), (0.5,)))
    ])

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(Normalize((0.5,), (0.5,))),

    ])

    return train_trf, test_trf


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Discriminator(Module):
    def __init__(self, nc=1, ndf=64, n_cls=10):
        super(Discriminator, self).__init__()
        # input is (nc) x 32 x 32
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf) x 16 x 16

        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*2) x 8 x 8

        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))  # state size. (ndf*4) x 4 x 4

        self.valid = nn.Sequential(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                                   nn.Sigmoid())   # state size. 1x1x1

        self.classify = nn.Sequential(nn.Conv2d(ndf * 4, n_cls, 4, 1, 0, bias=False),
                                      nn.Softmax(dim=1))  # state size. n_clsx1x1

        self.apply(weights_init)

    def forward(self, x):
        o1 = self.layer1(x)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        validator = self.valid(o3).squeeze(-1).squeeze(-1)
        classifier = self.classify(o3).squeeze(-1).squeeze(-1)
        return torch.cat((validator, classifier), dim=-1)


class Generator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(ngf * 8),
                                    nn.ReLU(True))  # state size. (ngf*8) x 4 x 4

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 4),
                                    nn.ReLU(True))  # state size. (ngf*2) x 8 x 8

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(ngf * 2),
                                    nn.ReLU(True))  # state size. (ngf*2) x 16 x 16

        self.out = nn.Sequential(nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                                 nn.Sigmoid())  # state size. (nc) x 32 x 32

        self.apply(weights_init)

    def forward(self, x):
        if len(x.size()) != 2:
            raise ValueError("Input must have 2 dim but found {}".format(x.shape))

        o1 = self.layer1(x.view(x.size(0), x.size(1), 1, 1))
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)

        return self.out(o3)


def parse_item_mnist_gan(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'data': img, 'target': np.float32(1.0), 'class': target}


def parse_item_mnist_ssgan(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    ext_y = np.zeros(2, dtype=np.int64)
    ext_y[0] = 1
    ext_y[1] = target
    return {'data': img, 'target': ext_y, 'class': target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=300, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--d_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Learning rate (Discriminator)')
    parser.add_argument('--g_wd', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--beta1', type=float, default=1e-4, help='Weight decay (Generator)')
    parser.add_argument('--g_net_features', type=int, default=64, help='Number of features (Generator)')
    parser.add_argument('--d_net_features', type=int, default=64, help='Number of featuresGenerator)')
    parser.add_argument('--latent_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--n_classes', type=int, default=10, help='Num of classes')
    parser.add_argument('--device', type=str, default="cuda", help='Use `cuda` or `cpu`')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--grid_shape', type=tuple, default=(24, 24), help='Shape of grid of generated images')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args




