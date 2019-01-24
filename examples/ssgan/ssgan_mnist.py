from torch.nn import BCELoss
from torch import optim, Tensor
from tensorboardX import SummaryWriter

from collagen.core import Module
from collagen.callbacks import OnGeneratorBatchFreezer, OnDiscriminatorBatchFreezer, BackwardCallback
from collagen.callbacks import ProgressbarVisualizer, TensorboardSynthesisVisualizer, GeneratorLoss
from collagen.data import DataProvider, ItemLoader, SSGANFakeSampler
from collagen.strategies import GANStrategy
from collagen.metrics import RunningAverageMeter, SSAccuracyMeter, SSValidityMeter
from collagen.data.utils import get_mnist, auto_detect_device
from collagen.logging import MeterLogging

from examples.ssgan.ex_utils import init_args, parse_item_mnist_ssgan, init_mnist_transforms
from examples.ssgan.ex_utils import Discriminator, Generator

device = auto_detect_device()


class SSDicriminatorLoss(Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.__loss_valid = BCELoss()
        self.__loss_cls = BCELoss()
        self.__alpha = alpha

    def forward(self, pred: Tensor, target: Tensor):
        pred_valid = pred[:, 0]
        pred_cls = pred[:, 1:]

        target_valid = target[:, 0]
        target_cls = target[:, 1:]

        # target_cls = target_cls.type(torch.int64)
        loss_valid = self.__loss_valid(pred_valid, target_valid)
        loss_cls = self.__loss_cls(pred_cls, target_cls)

        return self.__alpha * loss_valid + (1 - self.__alpha) * loss_cls


if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "ssgan"

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Data
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = SSDicriminatorLoss(alpha=0.5).to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = GeneratorLoss(d_network=d_network, d_loss=BCELoss()).to(device)

    # Data provider
    item_loaders = dict()

    item_loaders['real'] = ItemLoader(meta_data=train_ds,
                                      transform=init_mnist_transforms()[1],
                                      parse_item_cb=parse_item_mnist_ssgan,
                                      batch_size=args.bs, num_workers=args.num_threads)

    item_loaders['fake'] = SSGANFakeSampler(g_network=g_network,
                                            batch_size=args.bs,
                                            latent_size=args.latent_size,
                                            n_classes=args.n_classes)

    data_provider = DataProvider(item_loaders)

    # Callbacks
    g_callbacks = (RunningAverageMeter(prefix="G", name="loss"),
                   OnGeneratorBatchFreezer(modules=d_network))
    d_callbacks = (RunningAverageMeter(prefix="D", name="loss"),
                   SSValidityMeter(threshold=0.5, sigmoid=False, prefix="D", name="ss_acc"),
                   SSAccuracyMeter(prefix="D", name="ss_valid"),
                   BackwardCallback(retain_graph=True),
                   OnDiscriminatorBatchFreezer(modules=g_network))
    st_callbacks = (ProgressbarVisualizer(update_freq=1),
                    MeterLogging(writer=summary_writer),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake'], writer=summary_writer, grid_shape=args.grid_shape))

    # Strategy
    num_samples_dict = {'real': 1, 'fake': 1}
    ssgan = GANStrategy(data_provider=data_provider,
                        g_loader_names=('fake'), d_loader_names=('real', 'fake'),
                        g_criterion=g_crit, d_criterion=d_crit,
                        g_model=g_network, d_model=d_network,
                        g_optimizer=g_optim, d_optimizer=d_optim,
                        g_data_key='latent', d_data_key=('data', 'data'),
                        g_target_key='target', d_target_key='target',
                        g_callbacks=g_callbacks, d_callbacks=d_callbacks,
                        callbacks=st_callbacks,
                        n_epochs=args.n_epochs, device=args.device,
                        num_samples_dict=num_samples_dict)

    ssgan.run()
