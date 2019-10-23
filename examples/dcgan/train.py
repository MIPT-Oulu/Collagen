import os
from multiprocessing import Value
import torch
import yaml
from collagen.callbacks.logging.loggers import FakeScalarMeterLogger
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import BCELoss
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import torch.backends.cudnn as cudnn
import numpy as np
import random
from collagen.callbacks.logging.visualization import ImageSamplingVisualizer, FakeImageSamplingVisualizer
from collagen.core import Trainer
from collagen.core.utils import auto_detect_device, init_dist_env
from collagen.data.data_provider import gan_data_provider
from collagen.data.utils.datasets import get_mnist
from collagen.callbacks import ScalarMeterLogger, RunningAverageMeter
from collagen.losses.gan import GeneratorLoss
from collagen.strategies import DualModelStrategy, MultiModelStrategy
from model import Discriminator, Generator
from utils import init_args, parse_item_mnist_gan, init_mnist_transforms
import apex
from apex.parallel import DistributedDataParallel as DDP_APEX
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from torch.nn.parallel import DistributedDataParallel as DDP


def main(args):
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
    if args.distributed:

        mp.spawn(worker_process, nprocs=ngpus_per_node, args=(ngpus_per_node,
                                                              sampling_config, strategy_config, args))
    else:
        worker_process(args.gpu, ngpus_per_node, sampling_config,  strategy_config, args)


def worker_process(gpu, ngpus_per_node,  sampling_config, strategy_config, args):
    # get dataset
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(gpu))
    if args.gpu is not None:
        print('Using GPU: ', args.gpu)
    if args.distributed:
        args.rank = int(os.environ['RANK']) * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank,
                                init_method='env://')
        print('Distributed Init done on GPU:', args.gpu)
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        # discriminator
        d_network = Discriminator(nc=args.n_channels, ndf=args.d_net_features).to(gpu)
        if args.use_apex:
            d_network = apex.parallel.convert_syncbn_model(d_network)
        d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))

        # generator
        g_network = Generator(nc=args.n_channels, nz=args.latent_size, ngf=args.g_net_features).to(gpu)
        if args.use_apex:
            g_network = apex.parallel.convert_syncbn_model(g_network)
        g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
        if args.use_apex:
            [d_network, g_network], [d_optim, g_optim] = amp.initialize([d_network, g_network], [d_optim, g_optim],
                                                                        opt_level='O1')
            d_network = DDP_APEX(d_network, delay_allreduce=True)
            g_network = DDP_APEX(g_network, delay_allreduce=True)
        else:
            # DDP does not have delay_allreduce functionality, as a result all the process needs to have same forward
            # passing, i.e. if one network forward passes in one process and in another process it does not, the
            # the system will hang as there is nothing to reduce by backend. For example, I used visualizer for for
            # gpu 0, but no visualization for gpu 1. For visualization, I had to forward pass data through one network,
            # so the backend tried to reduce it, found nothing on gpu 1 and the process hanged infinitely.
            d_network = DDP(d_network, device_ids=[gpu])
            g_network = DDP(g_network, device_ids=[gpu])
    else:
        # Initializing Discriminator
        d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
        d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
        # Initializing Generator
        g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
        g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
        args.workers = 0
        if args.use_apex:
            [d_network, g_network], [d_optim, g_optim] = amp.initialize([d_network, g_network], [d_optim, g_optim],
                                                                        opt_level='O1')

    # criterions
    d_crit = torch.nn.BCEWithLogitsLoss().to(args.gpu)

    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(args.gpu)

    # Initializing the data provider
    item_loaders = dict()
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, args.latent_size, init_mnist_transforms(),
                                      parse_item_mnist_gan, args)

    if args.gpu == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment='_' + comment+'gpu_'+str(args.gpu))
        # Setting up the callbacks
        #st_callbacks = (ScalarMeterLogger(writer=summary_writer),)
        train_callbacks = ()
        st_callbacks = (ScalarMeterLogger(writer=summary_writer),
                        ImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                          writer=summary_writer,
                                          grid_shape=(args.grid_shape, args.grid_shape)))
        val_cb = ()
        # st_callbacks = ()
        # val_cb = ()
    else:
        st_callbacks = (FakeScalarMeterLogger(writer=None),
                        FakeImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                                writer=None,
                                                grid_shape=(args.grid_shape, args.grid_shape)))
        val_cb = ()
        train_callbacks = ()
    # visiualization should not be called after every stage, st strategy call backs are called after the completion of
    # stage, so instead of putting the visualizer there,
    #st_callbacks = ()
    # Trainers


    d_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["D"].keys()),
                        val_loader_names=None,
                        module=d_network, optimizer=d_optim, loss=d_crit, use_apex=args.use_apex,
                        distributed=args.distributed,
                        train_callbacks=train_callbacks)

    g_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["G"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["G"].keys()),
                        val_callbacks=val_cb,
                        train_callbacks=train_callbacks,
                        module=g_network, optimizer=g_optim, loss=g_crit, use_apex=args.use_apex,
                        distributed=args.distributed)

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


if __name__ == '__main__':
    # parse arguments
    t = time.time()
    args = init_args()
    init_dist_env()
    device = auto_detect_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
    print('Required Time ', time.time() - t )