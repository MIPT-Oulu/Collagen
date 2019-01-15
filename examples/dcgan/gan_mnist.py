import torch
from torch.nn import BCELoss
from torch import optim

from collagen.core import Callback, Session
from collagen.data import DataProvider, ItemLoader, GANFakeSampler
from collagen.strategies import GANStrategy
from collagen.metrics import RunningAverageMeter, AccuracyThresholdMeter
from collagen.data.utils import get_mnist
from examples.dcgan.ex_utils import init_args, parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.ex_utils import Discriminator, Generator

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

class BackwardCallback(Callback):
    def __init__(self, retain_graph=True, create_graph=False):
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph
    def on_backward_begin(self, session: Session, **kwargs):
        session.set_backward_param(retain_graph=self.__retain_graph, create_graph=self.__create_graph)


if __name__ == "__main__":
    args = init_args()

    # Data
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit)

    # Data provider
    item_loaders = dict()

    item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                      transform=init_mnist_transforms()[1],
                                      parse_item_cb=parse_item_mnist_gan,
                                      batch_size=args.bs, num_workers=args.num_threads)

    item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                          batch_size=args.bs,
                                          latent_size=args.latent_size)

    data_provider = DataProvider(item_loaders)

    # Callbacks
    g_callbacks = (RunningAverageMeter(prefix="g"),)
    d_callbacks = (RunningAverageMeter(prefix="d"),
                   AccuracyThresholdMeter(threshold=0.5, sigmoid=False, prefix="d", name="acc"),
                   BackwardCallback(retain_graph=True))

    # Strategy
    num_samples_dict = {'real': 1, 'fake': 6}
    dcgan = GANStrategy(data_provider=data_provider,
                        g_loader_names=('fake'), d_loader_names=('real', 'fake'),
                        g_criterion=g_crit, d_criterion=d_crit,
                        g_model=g_network, d_model=d_network,
                        g_optimizer=g_optim, d_optimizer=d_optim,
                        g_data_key='latent', d_data_key=('data', 'data'),
                        g_target_key='target', d_target_key='target',
                        g_callbacks=g_callbacks, d_callbacks=d_callbacks,
                        n_epochs=args.n_epochs, device=args.device,
                        num_samples_dict=num_samples_dict)

    dcgan.run()
