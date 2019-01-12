import torch
from torch.nn import BCELoss
from torch import optim
from tqdm import tqdm

from collagen.data import DataProvider, ItemLoader, GANFakeSampler, GaussianNoiseSampler
from collagen.core import Session
from collagen.strategies import Strategy, GANStrategy
from collagen.data.utils import get_mnist
from ex_utils import init_args, parse_item_mnist_gan, init_mnist_transforms
from ex_utils import Discriminator, Generator

# from . import init_args, parse_item_mnist_gan, init_mnist_transforms
# from . import Discriminator, Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeneratorLoss(torch.nn.Module):
    def __init__(self, d_network, d_loss):
        super(GeneratorLoss, self).__init__()
        self.__d_network = d_network
        self.__d_loss = d_loss

    def forward(self, img: torch.Tensor, target: torch.Tensor):
        output = self.__d_network(img)
        loss = self.__d_loss(output, target)
        return loss


if __name__ == "__main__":
    args = init_args()

    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit)

    d_session = Session(module=d_network, optimizer=d_optim, loss=d_crit)
    g_session = Session(module=g_network, optimizer=g_optim, loss=g_crit)

    d_item_loaders = dict()
    g_item_loaders = dict()
    g_item_loaders['fake'] = GaussianNoiseSampler(batch_size=args.bs, latent_size=args.latent_size)

    d_item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                        transform=init_mnist_transforms()[1],
                                        parse_item_cb=parse_item_mnist_gan,
                                        batch_size=args.bs, num_workers=args.num_threads)

    d_item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                            batch_size=args.bs,
                                            latent_size=args.latent_size)

    g_data_provider = DataProvider(g_item_loaders)
    d_data_provider = DataProvider(d_item_loaders)
    dcgan = GANStrategy(g_data_provider=g_data_provider, d_data_provider=d_data_provider,
                        g_loader_names=('fake'), d_loader_names=('real', 'fake'),
                        g_criterion=g_crit, d_criterion=d_crit,
                        g_model=g_network, d_model=g_network,
                        g_optimizer=g_optim, d_optimizer=d_optim,
                        g_data_key=('img'), d_data_key=('img', ''),
                        g_target_key=('img'), d_target_key=(),
                        g_num_samples=(1), d_num_samples=(1, 1),
                        # g_callbacks=None, d_callbacks=None,
                        n_epochs=args.n_epochs, device=args.device)

    dcgan.run()
