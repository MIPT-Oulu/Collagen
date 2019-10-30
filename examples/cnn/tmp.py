import random

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

import yaml
from torch.utils.tensorboard import SummaryWriter
from collagen.core.utils import auto_detect_device
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


def worker_process(gpu, ngpus_per_node, kfold_train_losses, kfold_val_losses, kfold_val_accuracies,
                   classes, item_loaders, sampling_config, df_train, df_val, args):
    args.gpu = gpu  # this line of code is not redundant
    if args.gpu is not None:
        print('Using GPU: ', args.gpu)
    if args.distributed:
        args.rank = int(os.environ['RANK']) * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
        print('Distributed Init done')

    log_dir = args.log_dir
    comment = args.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    criterion = torch.nn.CrossEntropyLoss()
    if args.distributed:
        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes), n_channels=args.n_channels).to(gpu)
        model = apex.parallel.convert_syncbn_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=None, keep_batchnorm_fp32=None)
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = DDP(model, delay_allreduce=True)
        cudnn.benchmark = True

    else:
        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes), n_channels=args.n_channels).to(gpu)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        args.workers = 0

    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        if args.distributed:
            item_loaders[f'mnist_{stage}'] = DistributedItemLoader(meta_data=df,
                                                                   transform=init_mnist_cifar_transforms(
                                                                       args.n_channels,
                                                                       stage),
                                                                   parse_item_cb=parse_item_mnist,
                                                                   args=args)
        else:
            item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                        transform=init_mnist_cifar_transforms(
                                                            args.n_channels,
                                                            stage),
                                                        parse_item_cb=parse_item_mnist,
                                                        batch_size=args.batch_size, num_workers=args.workers,
                                                        shuffle=True if stage == "train" else False)
    data_provider = DataProvider(item_loaders)
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 AccuracyMeter(prefix="train", name="acc"))

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               AccuracyMeter(prefix="eval", name="acc"),
               ScalarMeterLogger(writer=summary_writer),
               ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))

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
    kfold_train_losses.append(train_cbs[0].current())
    kfold_val_losses.append(val_cbs[0].current())
    kfold_val_accuracies.append(val_cbs[1].current())


def main(args):
    if args.dataset == 'cifar10':
        train_ds, classes = get_cifar10(data_folder=args.save_data, train=True)
        args.n_channels = 3
    elif args.dataset == 'mnist':
        train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
        args.n_channels = 1
    else:
        raise ValueError('Not support dataset {}'.format(args.dataset))

    if args.distributed:
        args.world_size = int(os.environ['WORLD_SIZE'])

    # Tensorboard visualization
    kfold_train_losses = []
    kfold_val_losses = []
    kfold_val_accuracies = []

    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    for fold_id, (df_train, df_val) in enumerate(splitter):
        item_loaders = dict()
        # number of gpus
        ngpus_per_node = torch.cuda.device_count()

        if args.distributed:
            mp.spawn(worker_process, nprocs=ngpus_per_node, args=(ngpus_per_node, kfold_train_losses, kfold_val_losses,
                                                                  kfold_val_accuracies, classes, item_loaders,
                                                                  sampling_config, df_train,
                                                                  df_val, args))
        else:
            worker_process(args.gpu, ngpus_per_node, kfold_train_losses, kfold_val_losses,
                           kfold_val_accuracies, classes, item_loaders, sampling_config, df_train, df_val, args)

    print("k-fold training loss: {}".format(np.asarray(kfold_train_losses).mean()))
    print("k-fold validation loss: {}".format(np.asarray(kfold_val_losses).mean()))
    print("k-fold validation accuracy: {}".format(np.asarray(kfold_val_accuracies).mean()))


if __name__ == "__main__":
    # parse arguments
    args = init_args()

    device = auto_detect_device()
    # initialize distributed environment if necessary
    if args.distributed:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '22222'
        os.environ['WORLD_SIZE'] = '2'  # world size is the total computing devices number of node  * gpus per node
        os.environ['RANK'] = '0'

    # control randomness
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
