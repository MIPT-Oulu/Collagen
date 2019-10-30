import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import time
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from collagen.core.utils import kick_off_launcher, convert_according_to_args
from collagen.data import FoldSplit, ItemLoader
from collagen.data import DistributedItemLoader, DataProvider
from collagen.data.utils.datasets import get_mnist, get_cifar10

from collagen.callbacks import ScalarMeterLogger
from collagen.callbacks import RunningAverageMeter, AccuracyMeter
from collagen.callbacks import ModelSaver

from collagen.strategies import Strategy
from utils import SimpleConvNet
from utils import init_mnist_cifar_transforms, init_args
import os
import matplotlib.pyplot as plt


def ddp2normal(sd):
    keys = sd.keys()
    newKeys = list()
    for key in keys:
        newKeys.append((key[7:], key))
    for (new, old) in newKeys:
        sd[new] = sd.pop(old)
    return sd
if __name__ == "__main__":
    device = torch.device('cuda:0')
    directory = 'saved_models'
    images = 'images'
    auc_list =[]
    counter = 0
    args = init_args()
    device = torch.device('cuda')
    model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=10, n_channels=args.n_channels).to(device)

    sd = torch.load('snapshots/model_0006_20191025_141539_eval.loss_2.384.pth')
    sd2 = model.state_dict()
    sd3 = ddp2normal(sd)
    model.load_state_dict(sd3)

    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                    ])
    test_ds = datasets.MNIST(root=args.save_data, train=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    auc = 0
    acc = 0.0

    acc_list = list()
    for i, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            out = model(data)
            out = out.detach()
            data2 = out.argmax(dim=-1)
            target = target.detach()

            correct = torch.sum((data2 == target)*1.0).cpu().numpy()
            acc = correct / (len(data2))
            acc_list.append(acc)

    print(np.mean(acc_list))






