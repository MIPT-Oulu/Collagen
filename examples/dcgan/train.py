import yaml
from torch.nn import BCELoss
from torch import optim
from tensorboardX import SummaryWriter

from collagen.core import Trainer
from collagen.callbacks import  TensorboardSynthesisVisualizer, GeneratorLoss
from collagen.data.utils import gan_data_provider
from collagen.core.utils import auto_detect_device
from collagen.strategies import GANStrategy
from collagen.data.utils import get_mnist
from collagen.logging import MeterLogging

from examples.dcgan.utils import init_args, parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.model import Discriminator, Generator

device = auto_detect_device()


if __name__ == "__main__":
    # Setup configs
    args = init_args()
    summary_writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
    g_optim = optim.Adam(g_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(device)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)



    # Initializing the data provider
    item_loaders = dict()
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    data_provider = gan_data_provider(g_network, item_loaders,
                                      train_ds, classes, args.latent_size, init_mnist_transforms(),
                                      parse_item_mnist_gan, args.bs, args.num_threads, device)
    # Setting up the callbacks

    st_callbacks = (MeterLogging(writer=summary_writer),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake'],
                                                   writer=summary_writer,
                                                   grid_shape=args.grid_shape))

    # Trainers
    d_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["D"].keys()),
                        val_loader_names=None,
                        module=d_network, optimizer=d_optim, loss=d_crit)

    g_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["G"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["G"].keys()),
                        module=g_network, optimizer=g_optim, loss=g_crit)

    # Strategy
    dcgan = GANStrategy(data_provider=data_provider, data_sampling_config=sampling_config,
                        d_trainer=d_trainer, g_trainer=g_trainer,
                        n_epochs=args.n_epochs,
                        callbacks=st_callbacks,
                        device=args.device)

    dcgan.run()
