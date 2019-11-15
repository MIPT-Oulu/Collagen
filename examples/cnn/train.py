import random
import warnings

import numpy as np
import torch
import torch.utils.data.distributed
import time
import yaml
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from collagen.core.utils import kick_off_launcher, init_dist_env, configure_model_optimizer
from collagen.data import FoldSplit, ItemLoader
from collagen.data import DataProvider
from collagen.data.utils.datasets import get_mnist, get_cifar10

from collagen.callbacks import ScalarMeterLogger
from collagen.callbacks import RunningAverageMeter, AccuracyMeter
from collagen.callbacks import ModelSaver

from collagen.strategies import Strategy
from examples.cnn.utils import SimpleConvNet
from examples.cnn.utils import init_mnist_cifar_transforms
from examples.cnn.arguments import init_args


def parse_item_mnist(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {data_key: img, target_key: target}


def worker_process(local_rank, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not args.ngpus_per_node == 0:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    if args.distributed:
        lr_m = float(args.batch_size * args.world_size) / 256.
    else:
        lr_m = 1.0

    # load some yml files for sampling and strategy configuration
    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f, Loader=yaml.FullLoader)
    if args.mms:
        with open("strategy.yml", "r") as f:
            strategy_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        strategy_config = None

    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_data, train=False)
    model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes), n_channels=args.n_channels).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr * lr_m, weight_decay=args.wd)

    model, optimizer = configure_model_optimizer(args=args,
                                                 local_rank=local_rank,
                                                 network=model,
                                                 optim=optimizer)

    item_loaders = dict()
    for stage, df in zip(['train', 'eval'], [train_ds, test_ds]):
        item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                    transform=init_mnist_cifar_transforms(1, stage),
                                                    parse_item_cb=parse_item_mnist,
                                                    distributed=args.distributed,
                                                    shuffle=not args.distributed,
                                                    pin_memory=args.distributed,
                                                    local_rank=local_rank,
                                                    world_size=args.world_size,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.workers)

    data_provider = DataProvider(item_loaders)
    if local_rank == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment='_' + comment + 'gpu_' + str(local_rank))
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
                        device=torch.device('cuda:{}'.format(local_rank)),
                        distributed=args.distributed,
                        use_apex=args.use_apex)

    strategy.run()


if __name__ == '__main__':
    """When the file is run"""
    t = time.time()
    # parse arguments
    args = init_args()
    if args.suppress_warning:
        warnings.filterwarnings("ignore")

    if args.shell_launch:
        if args.distributed:
            dist.init_process_group(backend=args.dist_backend, init_method='env://')
            init_dist_env()
        worker_process(args.local_rank, args)
    else:
        # kick off the main function
        kick_off_launcher(args=args,
                          worker_process=worker_process)

    print('Execution Time ', (time.time() - t), ' Seconds')
