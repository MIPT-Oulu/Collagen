from torch.nn import BCELoss
from torch import optim
from tensorboardX import SummaryWriter

from collagen.core import Module
from collagen.callbacks import Callback, OnGeneratorBatchFreezer, OnDiscriminatorBatchFreezer, BackwardCallback
from collagen.callbacks import ProgressbarVisualizer, TensorboardSynthesisVisualizer, GeneratorLoss
from collagen.data import DataProvider, ItemLoader, GANFakeSampler
from collagen.strategies import GANStrategy
from collagen.metrics import RunningAverageMeter, AccuracyThresholdMeter
from collagen.data.utils import get_mnist, auto_detect_device
from collagen.logging import MeterLogging

from examples.dcgan.ex_utils import init_args, parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.ex_utils import Discriminator, Generator

device = auto_detect_device()


class CalmDownDiscCallback(Callback):
    def __init__(self, d_module: Module):
        super().__init__(type="gan_trick")
        self.__d_module = d_module

    def on_minibatch_begin(self, **kwargs):
        pass


if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "dcgan"
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Data
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features, ngpu=args.ngpu, drop_rate=0.5).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features, ngpu=args.ngpu).to(device)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=BCELoss().to(device)).to(device)

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
    g_callbacks = (RunningAverageMeter(prefix="g", name="loss"),
                   OnGeneratorBatchFreezer(modules=d_network))

    d_callbacks = (RunningAverageMeter(prefix="d", name="loss"),
                   AccuracyThresholdMeter(threshold=0.5, sigmoid=False, prefix="d", name="acc"),
                   BackwardCallback(retain_graph=True),
                   OnDiscriminatorBatchFreezer(modules=g_network))

    st_callbacks = (ProgressbarVisualizer(update_freq=1),
                    MeterLogging(writer=summary_writer),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake'],
                                                   grid_shape=args.grid_shape,
                                                   writer=summary_writer))

    # Strategy
    num_samples_dict = {'real': 1, 'fake': 1}
    dcgan = GANStrategy(data_provider=data_provider,
                        g_loader_names='fake', d_loader_names=('real', 'fake'),
                        g_criterion=g_crit, d_criterion=d_crit,
                        g_model=g_network, d_model=d_network,
                        g_optimizer=g_optim, d_optimizer=d_optim,
                        g_data_key='latent', d_data_key=('data', 'data'),
                        g_target_key='target', d_target_key='target',
                        g_callbacks=g_callbacks, d_callbacks=d_callbacks,
                        callbacks=st_callbacks,
                        n_epochs=args.n_epochs, device=args.device,
                        num_samples_dict=num_samples_dict)

    dcgan.run()
