from collections import OrderedDict
import tqdm
import torch
import yaml
from torch.nn import BCELoss
from torch import optim
from tensorboardX import SummaryWriter

from collagen.core import Callback, Session, Trainer
from collagen.callbacks import OnGeneratorBatchFreezer, OnDiscriminatorBatchFreezer, ClipGradCallback
from collagen.callbacks import ProgressbarVisualizer, TensorboardSynthesisVisualizer, GeneratorLoss, OnSamplingFreezer
from collagen.data import DataProvider, ItemLoader, GANFakeSampler, GaussianNoiseSampler, SSFoldSplit
from collagen.data.utils import auto_detect_device
from collagen.strategies import GANStrategy
from collagen.metrics import RunningAverageMeter
from collagen.data.utils import get_mnist
from collagen.logging import MeterLogging

from examples.dcgan.ex_utils import init_args, parse_item_mnist_gan, init_mnist_transforms
from examples.dcgan.ex_utils import Discriminator, Generator

device = auto_detect_device()


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

    def on_batch_end(self, strategy: GANStrategy, epoch: int, progress_bar: tqdm, callbacks: Callback, **kwargs):
        self.__count += 1
        if self._check_freq():
            list_metrics_desc = []
            postfix_progress = OrderedDict()
            for cb in callbacks:
                if cb.get_type() == "meter":
                    list_metrics_desc.append(str(cb))
                    postfix_progress[cb.get_name()] = f'{cb.current():.03f}'

            progress_bar.set_postfix(ordered_dict=postfix_progress, refresh=True)


if __name__ == "__main__":
    # Setup configs
    args = init_args()
    n_classes = 10

    # Tensorboard visualization
    log_dir = args.log_dir
    comment = args.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    # Data provider
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    splitter = SSFoldSplit(train_ds, n_ss_folds=3, n_folds=5, target_col="target", random_state=args.seed,
                           labeled_train_size_per_class=1000, unlabeled_train_size_per_class=2000,
                           equal_target=True, equal_unlabeled_target=True, shuffle=True)

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

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

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    item_loaders['real_train'] = ItemLoader(meta_data=train_labeled_data,
                                            transform=init_mnist_transforms()[1],
                                            parse_item_cb=parse_item_mnist_gan,
                                            batch_size=args.bs, num_workers=args.num_threads)

    item_loaders['fake'] = GANFakeSampler(g_network=g_network,
                                                batch_size=args.bs,
                                                latent_size=args.latent_size)

    item_loaders['real_eval'] = ItemLoader(meta_data=val_labeled_data,
                                           transform=init_mnist_transforms()[1],
                                           parse_item_cb=parse_item_mnist_gan,
                                           batch_size=args.bs, num_workers=args.num_threads)



    item_loaders['noise'] = GaussianNoiseSampler(batch_size=args.bs,
                                                 latent_size=args.latent_size,
                                                 device=device, n_classes=n_classes)

    data_provider = DataProvider(item_loaders)

    # Callbacks
    g_callbacks_train = (RunningAverageMeter(prefix="train/G", name="loss"),
                         OnGeneratorBatchFreezer(modules=d_network),
                         ClipGradCallback(g_network, mode="norm", max_norm=0.1, norm_type=2))

    g_callbacks_eval = RunningAverageMeter(prefix="eval/G", name="loss")

    d_callbacks_train = (RunningAverageMeter(prefix="train/D", name="loss"),
                         OnDiscriminatorBatchFreezer(modules=g_network))

    d_callbacks_eval = (RunningAverageMeter(prefix="eval/D", name="loss"),)

    st_callbacks = (ProgressbarVisualizer(update_freq=1),
                    MeterLogging(writer=summary_writer),
                    OnSamplingFreezer(modules=(g_network, d_network)),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake'],
                                                   writer=summary_writer,
                                                   grid_shape=args.grid_shape))

    # Sessions
    d_session = Session(module=d_network,
                        optimizer=d_optim,
                        loss=d_crit)

    g_session = Session(module=g_network,
                        optimizer=g_optim,
                        loss=g_crit)

    # Trainers
    d_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["D"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["D"].keys()),
                        session=d_session,
                        train_callbacks=d_callbacks_train,
                        val_callbacks=d_callbacks_eval)

    g_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["G"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["G"].keys()),
                        session=g_session,
                        train_callbacks=g_callbacks_train,
                        val_callbacks=g_callbacks_eval)

    # Strategy
    dcgan = GANStrategy(data_provider=data_provider,
                        data_sampling_config=sampling_config,
                        d_trainer=d_trainer,
                        g_trainer=g_trainer,
                        n_epochs=args.n_epochs,
                        callbacks=st_callbacks,
                        device=args.device)

    dcgan.run()
