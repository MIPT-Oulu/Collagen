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
from torchvision import transforms
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

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def parse_item_mnist(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {data_key: img, target_key: target}


def worker_process(gpu, ngpus,  sampling_config, strategy_config, args):
    args.gpu = gpu  # this line of code is not redundant
    if args.distributed:
        lr_m = float(args.batch_size*args.world_size)/256.
    else:
        lr_m = 1.0
    criterion = torch.nn.CrossEntropyLoss().to(gpu)
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_data, train=False)
    model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes), n_channels=args.n_channels).to(gpu)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr*lr_m, weight_decay=args.wd)

    args, model, optimizer = convert_according_to_args(args=args,
                                                       gpu=gpu,
                                                       ngpus=ngpus,
                                                       network=model,
                                                       optim=optimizer)

    # v = 0.3
    # l = len(train_ds)
    # lim = int(l*(1-v))
    # df_train = train_ds[:lim]
    # df_val = train_ds[lim:]
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
    item_loaders = dict()
    for stage, df in zip(['train', 'eval'], [train_ds, test_ds]):
        if args.distributed:
            item_loaders[f'mnist_{stage}'] = DistributedItemLoader(meta_data=df,
                                                                   transform=init_mnist_cifar_transforms(1, stage),
                                                                   parse_item_cb=parse_item_mnist,
                                                                   args=args)
        else:
            item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                        transform=init_mnist_cifar_transforms(1, stage),
                                                        parse_item_cb=parse_item_mnist,
                                                        batch_size=args.batch_size, num_workers=args.workers,
                                                        shuffle=True if stage == "train" else False)
    data_provider = DataProvider(item_loaders)
    if args.gpu == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment='_' + comment + 'gpu_' + str(args.gpu))
        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     AccuracyMeter(prefix="train", name="acc"))

        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   AccuracyMeter(prefix="eval", name="acc"),
                   ScalarMeterLogger(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))
    else:
        train_cbs = ()
        val_cbs = ()

    strategy = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config['train']['data_provider'].keys()),
                        val_loader_names=tuple(sampling_config['eval']['data_provider'].keys()),
                        data_sampling_config=sampling_config,
                        loss=criterion,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optimizer,
                        train_callbacks=train_cbs,
                        val_callbacks=val_cbs,
                        device=torch.device('cuda:{}'.format(args.gpu)),
                        distributed=args.distributed)

    strategy.run()


if __name__ == '__main__':
    """When the file is run"""
    t = time.time()
    # parse arguments
    args = init_args()
    # kick off the main function
    kick_off_launcher(args, worker_process)
    print('Execution Time ', (time.time() - t), ' Seconds')
