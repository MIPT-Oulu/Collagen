from torch.nn import BCELoss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
import os
import hydra

# You can be "straightforward" as below
# from collagen import *
# Or just be a nerd
from collagen.core import Session
from collagen.strategies import Strategy
from collagen.core.utils import auto_detect_device
from collagen.data import get_mnist, gan_data_provider
from collagen.callbacks import ScalarMeterLogger, ImageSamplingVisualizer, RunningAverageMeter
from collagen.losses import GeneratorLoss

from examples.dcgan.utils import parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.model import Discriminator, Generator

device = auto_detect_device()


@hydra.main(config_path='configs/config.yaml')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    log_dir = os.path.join(os.getcwd(), cfg.log_dir)
    summary_writer = SummaryWriter(log_dir=log_dir, comment=cfg.comment)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=cfg.d_net_features, drop=cfg.dropout).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=cfg.d_lr, weight_decay=cfg.d_wd, betas=(cfg.d_beta, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=cfg.latent_size, ngf=cfg.g_net_features).to(device)
    g_optim = optim.Adam(g_network.parameters(), lr=cfg.g_lr, weight_decay=cfg.g_wd, betas=(cfg.g_beta, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(device)

    # Initializing the data provider
    item_loaders = dict()
    data_dir = os.path.join(os.environ['PWD'], cfg.data_dir)
    train_ds, classes = get_mnist(data_folder=data_dir, train=True)
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, cfg.latent_size, init_mnist_transforms(),
                                      parse_item_mnist_gan, cfg.bs, cfg.num_threads, device)

    # Setting up the callbacks
    st_callbacks = (ScalarMeterLogger(writer=summary_writer),
                    ImageSamplingVisualizer(generator_sampler=item_loaders['fake'],
                                            transform=lambda x: (x+1.0)/2.0,
                                            writer=summary_writer,
                                            grid_shape=(cfg.grid_shape, cfg.grid_shape)))

    # Session
    d_session = Session(data_provider=data_provider,
                        train_loader_names=cfg.sampling.train.data_provider.D.keys(),
                        val_loader_names=None,
                        train_callbacks=RunningAverageMeter(prefix="train/D", name="loss"),
                        module=d_network, optimizer=d_optim, loss=d_crit)

    g_session = Session(data_provider=data_provider,
                        train_loader_names=cfg.sampling.train.data_provider.G.keys(),
                        val_loader_names=cfg.sampling.eval.data_provider.G.keys(),
                        train_callbacks=RunningAverageMeter(prefix="train/G", name="loss"),
                        val_callbacks=RunningAverageMeter(prefix="eval/G", name="loss"),
                        module=g_network, optimizer=g_optim, loss=g_crit)

    # Strategy
    dcgan = Strategy(data_provider=data_provider,
                     data_sampling_config=cfg.sampling,
                     strategy_config=cfg.strategy,
                     sessions=(d_session, g_session),
                     n_epochs=cfg.n_epochs,
                     callbacks=st_callbacks,
                     device=device)

    dcgan.run()


if __name__ == "__main__":
    main()
