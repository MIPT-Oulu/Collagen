import os
from multiprocessing import Value
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import BCELoss
import torch.distributed as dist
import torch.multiprocessing as mp

import torch.backends.cudnn as cudnn
import numpy as np
import random
from collagen.callbacks.logging.visualization import ImageSamplingVisualizer
from collagen.core import Trainer
from collagen.core.utils import auto_detect_device
from collagen.data.data_provider import gan_data_provider
from collagen.data.utils.datasets import get_mnist
from collagen.callbacks import ScalarMeterLogger, RunningAverageMeter
from collagen.losses.gan import GeneratorLoss
from collagen.strategies import DualModelStrategy, MultiModelStrategy
from model import Discriminator, Generator
from utils import init_args, parse_item_mnist_gan, init_mnist_transforms

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main(args):

    # get dataset
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    # set number of channels
    args.n_channels = 1
    if args.distributed:
        args.world_size = int(os.environ['WORLD_SIZE'])

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    if args.mms:
        with open("strategy.yml", "r") as f:
            strategy_config = yaml.load(f)
    else:
        strategy_config = None

    ngpus_per_node = torch.cuda.device_count()

    # empty itemloaders
    item_loaders = dict()
    if args.distributed:
        mp.spawn(worker_process, nprocs=ngpus_per_node, args=(ngpus_per_node, classes, item_loaders,
                                                              sampling_config, strategy_config, train_ds, args))
    else:
        worker_process(args.gpu, ngpus_per_node, classes, item_loaders, sampling_config,  strategy_config, train_ds,
                       args)


def worker_process(gpu, ngpus_per_node, classes, item_loaders,  sampling_config, strategy_config, train_ds, args):
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(gpu))
    if args.gpu is not None:
        print('Using GPU: ', args.gpu)
    if args.distributed:
        args.rank = int(os.environ['RANK']) * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
        print('Distributed Init done on GPU:', args.gpu)

    if args.gpu == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    if args.distributed:
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        # discriminator
        d_network = Discriminator(nc=args.n_channels, ndf=args.d_net_features).to(gpu)
        d_network = apex.parallel.convert_syncbn_model(d_network)
        d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))

        # generator
        g_network = Generator(nc=args.n_channels, nz=args.latent_size, ngf=args.g_net_features).to(gpu)
        g_network = apex.parallel.convert_syncbn_model(g_network)
        g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
        # g_network, g_optim = amp.initialize(g_network, g_optim, opt_level='O1')
        [d_network, g_network], [d_optim, g_optim] = amp.initialize([d_network, g_network], [d_optim, g_optim],
                                                                    opt_level='O1')
        d_network = DDP(d_network, delay_allreduce=True)
        g_network = DDP(g_network, delay_allreduce=True)
    else:
        # Initializing Discriminator
        d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
        d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
        # Initializing Generator
        g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
        g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
        args.workers = 0

    # criterions
    d_crit = torch.nn.BCEWithLogitsLoss()

    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit)


    # Initializing the data provider
    item_loaders = dict()
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, args.latent_size, init_mnist_transforms(),
                                      parse_item_mnist_gan, args)
    if args.gpu == 0:
        # Setting up the callbacks
        #st_callbacks = (ScalarMeterLogger(writer=summary_writer),)
        st_callbacks = (ScalarMeterLogger(writer=summary_writer),
                        ImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                                writer=summary_writer,
                                                grid_shape=(args.grid_shape, args.grid_shape)))
    else:
        st_callbacks = ()
    #st_callbacks = ()
    # Trainers
    d_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["D"].keys()),
                        val_loader_names=None,
                        module=d_network, optimizer=d_optim, loss=d_crit, distributed=args.distributed)

    g_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["G"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["G"].keys()),
                        module=g_network, optimizer=g_optim, loss=g_crit, distributed=args.distributed)

    # Strategy
    if args.mms:
        trainers = (g_trainer, d_trainer)
        dcgan = MultiModelStrategy(data_provider=data_provider,
                                   data_sampling_config=sampling_config,
                                   strategy_config=strategy_config,
                                   device=device,
                                   callbacks=st_callbacks,
                                   n_epochs=args.n_epochs,
                                   trainers=trainers,
                                   distributed=args.distributed)
    else:
        dcgan = DualModelStrategy(data_provider=data_provider, data_sampling_config=sampling_config,
                                  m0_trainer=d_trainer, m1_trainer=g_trainer, model_names=("D", "G"),
                                  n_epochs=args.n_epochs, callbacks=st_callbacks, device=device,
                                  distributed=args.distributed)

    dcgan.run()


if __name__== '__main__':
    # parse arguments
    args = init_args()


    device = auto_detect_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '22222'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '0'
    main(args)