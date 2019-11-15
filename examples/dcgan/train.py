import os
import torch
import yaml
from collagen.callbacks.logging.loggers import FakeScalarMeterLogger
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
import torch.distributed as dist
from collagen.callbacks.logging.visualization import ImageSamplingVisualizer, FakeImageSamplingVisualizer
from collagen.core import Trainer
from collagen.core.utils import configure_model_optimizer, kick_off_launcher, init_dist_env
from collagen.data.data_provider import gan_data_provider
from collagen.data.utils.datasets import get_mnist
from collagen.callbacks import ScalarMeterLogger
from collagen.losses.gan import GeneratorLoss
from collagen.strategies import DualModelStrategy, MultiModelStrategy
from examples.dcgan.model import Discriminator, Generator
from examples.dcgan.utils import parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.arguments import init_args
import numpy as np
import random


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

    # get MNIST dataset
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # discriminator
    d_network = Discriminator(nc=args.n_channels, ndf=args.d_net_features).to(device)
    # generator
    lr_m = float(args.batch_size * args.world_size) / 256.
    g_network = Generator(nc=args.n_channels, nz=args.latent_size, ngf=args.g_net_features).to(device)
    d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr * lr_m, betas=(args.beta1, 0.99))
    g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr * lr_m, betas=(args.beta1, 0.99))

    [d_network, g_network], [d_optim, g_optim] = configure_model_optimizer(args=args,
                                                                           local_rank=local_rank,
                                                                           network=[d_network, g_network],
                                                                           optim=[d_optim, g_optim])

    # criterions
    d_crit = torch.nn.BCEWithLogitsLoss().to(local_rank)
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(local_rank)

    # Initializing the data provider

    item_loaders = dict()
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, args.latent_size,
                                      transform=init_mnist_transforms(),
                                      parse_item=parse_item_mnist_gan,
                                      distributed=args.distributed,
                                      shuffle=not args.distributed,
                                      pin_memory=args.distributed,
                                      local_rank=local_rank,
                                      world_size=args.world_size,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers
                                      )

    # callbacks
    if local_rank == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment='_' + comment + 'gpu_' + str(local_rank))
        train_callbacks = ()
        st_callbacks = (ScalarMeterLogger(writer=summary_writer),
                        ImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                                writer=summary_writer,
                                                grid_shape=(args.grid_shape, args.grid_shape)))
        val_cb = ()
    else:
        st_callbacks = (FakeScalarMeterLogger(writer=None),
                        FakeImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                                    writer=None,
                                                    grid_shape=(args.grid_shape, args.grid_shape)))
        val_cb = ()
        train_callbacks = ()

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
    """When the file is run"""
    t = time.time()
    # parse arguments
    args = init_args()
    if args.shell_launch:
        if args.distributed:
            dist.init_process_group(backend=args.dist_backend, init_method='env://')
            init_dist_env()
        worker_process(args.local_rank, args)
    else:
        # kick off the main function
        kick_off_launcher(args, worker_process)
    print('Execution Time ', (time.time() - t), ' Seconds')
