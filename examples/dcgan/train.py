import os
import torch
import yaml
from collagen.callbacks.logging.loggers import FakeScalarMeterLogger
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import time
from collagen.callbacks.logging.visualization import ImageSamplingVisualizer, FakeImageSamplingVisualizer
from collagen.core import Trainer
from collagen.core.utils import convert_according_to_args, kick_off_launcher
from collagen.data.data_provider import gan_data_provider
from collagen.data.utils.datasets import get_mnist
from collagen.callbacks import ScalarMeterLogger
from collagen.losses.gan import GeneratorLoss
from collagen.strategies import DualModelStrategy, MultiModelStrategy
from model import Discriminator, Generator
from utils import init_args, parse_item_mnist_gan, init_mnist_transforms
import numpy as np
import random

def worker_process(gpu, ngpus,  sampling_config, strategy_config, args):
    """
    Parameters
    ----------
    gpu: int
        the id of current gpu
    ngpus: int
        total number of gpu(s)
    sampling_config: dict
        contains configuration for data sampling
    strategy_config: dict
        contains configuration for strategy
    args: Namespace
        parsed argument from argument parser

    Returns
    -------

    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # get MNIST dataset
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    args.gpu = gpu
    device = torch.device('cuda:{}'.format(gpu))
    if args.gpu is not None:
        print('Using GPU: ', args.gpu)

    # discriminator
    d_network = Discriminator(nc=args.n_channels, ndf=args.d_net_features).to(gpu)
    # generator
    lr_m = float(args.batch_size*args.world_size) / 256.
    g_network = Generator(nc=args.n_channels, nz=args.latent_size, ngf=args.g_net_features).to(gpu)
    d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr*lr_m, betas=(args.beta1, 0.99))
    g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr*lr_m, betas=(args.beta1, 0.99))

    args, [d_network, g_network], [d_optim, g_optim] = convert_according_to_args(args=args,
                                                                                  gpu=gpu,
                                                                                  ngpus=ngpus,
                                                                                  network=[d_network, g_network],
                                                                                  optim=[d_optim, g_optim])
    # criterions
    d_crit = torch.nn.BCEWithLogitsLoss().to(args.gpu)
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(args.gpu)

    # Initializing the data provider
    item_loaders = dict()
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, args.latent_size, init_mnist_transforms(),
                                      parse_item_mnist_gan, args)

    # callbacks
    if args.gpu == 0:
        log_dir = args.log_dir
        comment = args.comment
        summary_writer = SummaryWriter(log_dir=log_dir, comment='_' + comment+'gpu_'+str(args.gpu))
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
    # kick off the main function
    kick_off_launcher(args, worker_process)
    print('Execution Time ', (time.time() - t) , ' Seconds')