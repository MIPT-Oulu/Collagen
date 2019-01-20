from collections import OrderedDict
from typing import Tuple
import tqdm
import torch
from docutils.nodes import comment
from torch.nn import BCELoss
from torch import optim
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from collagen.core import Callback, Session
from collagen.data import DataProvider, ItemLoader, GANFakeSampler
from collagen.strategies import GANStrategy
from collagen.metrics import RunningAverageMeter, AccuracyThresholdMeter
from collagen.data.utils import get_mnist
from collagen.logging import MeterLogging
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
        loss = self.__d_loss(output, 1 - target)
        return loss


class BackwardCallback(Callback):
    def __init__(self, retain_graph=True, create_graph=False):
        super().__init__()
        self.__retain_graph = retain_graph
        self.__create_graph = create_graph

    def on_backward_begin(self, session: Session, **kwargs):
        session.set_backward_param(retain_graph=self.__retain_graph, create_graph=self.__create_graph)


class ProgressbarCallback(Callback):
    def __init__(self, update_freq=1):
        self.__type = "progressbar"
        super().__init__(type=self.__type)
        self.__count = 0
        self.__update_freq = update_freq
        if not isinstance(self.__update_freq, int) or self.__update_freq < 1:
            raise ValueError(
                "`update_freq` must be `int` and greater than 0, but found {} {}".format(type(self.__update_freq),
                                                                                         self.__update_freq))

    def _check_freq(self):
        return self.__count % self.__update_freq == 0

    def on_batch_end(self, strategy: GANStrategy, epoch: int, progress_bar: tqdm, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            for cb in strategy.get_callbacks_by_name("minibatch"):
                if cb.get_type() == "meter":
                    list_metrics_desc.append(str(cb))
                    postfix_progress[cb.get_name()] = f'{cb.current():.03f}'

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)


class GeneratorCallback(Callback):
    def __init__(self, writer, generator_sampler: GANFakeSampler, tag: str = "generated", log_dir: str = None,
                 comment: str = "", grid_shape: Tuple[int] = (6, 6)):
        super().__init__(type="visualizer")
        self.__generator_sampler = generator_sampler

        if len(grid_shape) != 2:
            raise ValueError("`grid_shape` must have 2 dim, but found {}".format(len(grid_shape)))

        self.__writer = writer

        self.__grid_shape = grid_shape
        self.__num_images = grid_shape[0] * grid_shape[1]
        self.__num_batches = self.__num_images // self.__generator_sampler.batch_size + 1
        self.__tag = tag

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        sampled_data = self.__generator_sampler.sample(self.__num_batches)
        images = []
        for i, dt in enumerate(sampled_data):
            if i < self.__num_images:
                images += list(torch.unbind(dt["data"], dim=0))
            else:
                break
        grid_images = make_grid(images[:self.__num_images], nrow=self.__grid_shape[0])
        self.__writer.add_images(self.__tag, img_tensor=grid_images, global_step=epoch, dataformats='CHW')


if __name__ == "__main__":
    args = init_args()
    log_dir = None
    comment = "dcgan"
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Data
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = BCELoss().to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=d_crit).to(device)

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
    g_callbacks = (RunningAverageMeter(prefix="g", name="loss"),)
    d_callbacks = (RunningAverageMeter(prefix="d", name="loss"),
                   AccuracyThresholdMeter(threshold=0.5, sigmoid=False, prefix="d", name="acc"),
                   BackwardCallback(retain_graph=True))
    st_callbacks = (ProgressbarCallback(update_freq=1),
                    MeterLogging(writer=summary_writer),
                    GeneratorCallback(generator_sampler=item_loaders['fake'], grid_shape=(4, 4),
                                      writer=summary_writer))

    # Strategy
    num_samples_dict = {'real': 1, 'fake': 30}
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
