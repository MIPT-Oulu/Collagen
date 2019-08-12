import yaml
from tensorboardX import SummaryWriter
from torch import optim, Tensor
from torch.nn import BCELoss, CrossEntropyLoss

from collagen.callbacks import ConfusionMatrixVisualizer
from collagen.callbacks import TensorboardSynthesisVisualizer, ClipGradCallback
from collagen.core import Module, Trainer
from collagen.core.utils import auto_detect_device, to_cpu
from collagen.data import DataProvider, ItemLoader, SSGANFakeSampler, SSFoldSplit
from collagen.data.utils.datasets import get_mnist
from collagen.callbacks.logging import ScalarMeterLogger
from collagen.callbacks.metrics import SSAccuracyMeter, SSValidityMeter
from collagen.strategies import GANStrategy
from examples.ssgan.utils import Discriminator, Generator
from examples.ssgan.utils import init_args, parse_item_mnist_ssgan, init_mnist_transforms

device = auto_detect_device()


class SSDicriminatorLoss(Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.__loss_valid = BCELoss()
        self.__loss_cls = CrossEntropyLoss()
        self.__alpha = alpha

    def forward(self, pred: Tensor, target: Tensor):
        pred_valid = pred[:, -1]
        pred_cls = pred[:, 0:-1]

        if len(target.shape) > 1 and target.shape[1] > 1:
            target_valid = target[:, -1]
            target_cls = target[:, 0:-1]

            loss_valid = self.__loss_valid(pred_valid, target_valid)
            decoded_target_cls = target_cls.argmax(dim=-1)
            loss_cls = self.__loss_cls(pred_cls, decoded_target_cls)

            _loss = self.__alpha * loss_valid + (1 - self.__alpha) * loss_cls
        else:
            target_valid = target
            _loss = self.__loss_valid(pred_valid, target_valid)
        return _loss


class SSGeneratorLoss(Module):
    def __init__(self, d_network, d_loss):
        super().__init__()
        self.__d_network = d_network
        self.__d_loss = d_loss

    def forward(self, img: Tensor, target: Tensor):
        # freeze_modules(modules=self.__d_network)
        output = self.__d_network(img)
        # freeze_modules(modules=self.__d_network, invert=True)
        output_valid = output[:, -1]
        if len(target.shape) > 1:
            target_fake = 1 - target[:, -1]
        else:
            target_fake = 1 - target
        loss = self.__d_loss(output_valid, target_fake)
        return loss


class SSConfusionMatrixVisualizer(ConfusionMatrixVisualizer):
    def __init__(self, writer, labels: list or None = None, tag="confusion_matrix"):
        super().__init__(writer=writer, labels=labels, tag=tag)

    def on_forward_end(self, output: Tensor, target: Tensor, **kwargs):
        if len(target.shape) > 1 and target.shape[1] > 1:
            pred_cls = output[:, 0:-1]
            target_cls = target[:, :-1]
            decoded_pred_cls = pred_cls.argmax(dim=-1)
            decoded_target_cls = target_cls.argmax(dim=-1)
            self._corrects += [self._labels[i] for i in to_cpu(decoded_target_cls, use_numpy=True).tolist()]
            self._predicts += [self._labels[i] for i in to_cpu(decoded_pred_cls, use_numpy=True).tolist()]


if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "ssgan"

    # Data provider
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    n_folds = 5
    splitter = SSFoldSplit(train_ds, n_ss_folds=3, n_folds=n_folds, target_col="target", random_state=args.seed,
                           labeled_train_size_per_class=100, unlabeled_train_size_per_class=200,
                           equal_target=True, equal_unlabeled_target=False, shuffle=True)

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    # Initializing Discriminator
    d_network = Discriminator(nc=1, ndf=args.d_net_features).to(device)
    d_optim = optim.Adam(d_network.parameters(), lr=args.d_lr, weight_decay=args.d_wd, betas=(args.beta1, 0.999))
    d_crit = SSDicriminatorLoss(alpha=0.5).to(device)

    # Initializing Generator
    g_network = Generator(nc=1, nz=args.latent_size, ngf=args.g_net_features).to(device)
    g_optim = optim.Adam(g_network.parameters(), lr=args.g_lr, weight_decay=args.g_wd, betas=(args.beta1, 0.999))
    g_crit = SSGeneratorLoss(d_network=d_network, d_loss=BCELoss()).to(device)

    item_loaders = dict()
    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    item_loaders["real_labeled_train"] = ItemLoader(meta_data=train_labeled_data,
                                                    transform=init_mnist_transforms()[1],
                                                    parse_item_cb=parse_item_mnist_ssgan,
                                                    batch_size=args.bs, num_workers=args.num_threads,
                                                    shuffle=True)

    item_loaders["real_unlabeled_train"] = ItemLoader(meta_data=train_unlabeled_data,
                                                      transform=init_mnist_transforms()[1],
                                                      parse_item_cb=parse_item_mnist_ssgan,
                                                      batch_size=args.bs, num_workers=args.num_threads,
                                                      shuffle=True)

    item_loaders["real_labeled_val"] = ItemLoader(meta_data=val_labeled_data,
                                                  transform=init_mnist_transforms()[1],
                                                  parse_item_cb=parse_item_mnist_ssgan,
                                                  batch_size=args.bs, num_workers=args.num_threads,
                                                  shuffle=False)

    item_loaders["real_unlabeled_val"] = ItemLoader(meta_data=val_unlabeled_data,
                                                    transform=init_mnist_transforms()[1],
                                                    parse_item_cb=parse_item_mnist_ssgan,
                                                    batch_size=args.bs, num_workers=args.num_threads,
                                                    shuffle=False)

    item_loaders['fake_unlabeled_gen'] = SSGANFakeSampler(g_network=g_network,
                                                          batch_size=args.bs,
                                                          latent_size=args.latent_size,
                                                          n_classes=args.n_classes)

    item_loaders['fake_unlabeled_latent'] = SSGANFakeSampler(g_network=g_network,
                                                             batch_size=args.bs,
                                                             latent_size=args.latent_size,
                                                             n_classes=args.n_classes)

    data_provider = DataProvider(item_loaders)

    # Callbacks
    g_callbacks_train = ClipGradCallback(g_network, mode="norm", max_norm=0.1, norm_type=2)

    d_callbacks_train = (SSValidityMeter(threshold=0.5, sigmoid=False, prefix="train/D", name="ss_valid"),
                         SSAccuracyMeter(prefix="train/D", name="ss_acc"))

    d_callbacks_eval = (SSValidityMeter(threshold=0.5, sigmoid=False, prefix="eval/D", name="ss_valid"),
                        SSAccuracyMeter(prefix="eval/D", name="ss_acc"),
                        SSConfusionMatrixVisualizer(writer=summary_writer,
                                                    labels=[str(i) for i in range(10)],
                                                    tag="eval/confusion_matrix"))

    st_callbacks = (ScalarMeterLogger(writer=summary_writer),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake_unlabeled_gen'],
                                                   writer=summary_writer,
                                                   grid_shape=args.grid_shape))

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    d_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["D"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["D"].keys()),
                        module=d_network,
                        optimizer=d_optim,
                        loss=d_crit,
                        train_callbacks=d_callbacks_train,
                        val_callbacks=d_callbacks_eval)

    g_trainer = Trainer(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"]["G"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"]["G"].keys()),
                        module=g_network,
                        optimizer=g_optim,
                        loss=g_crit,
                        train_callbacks=g_callbacks_train)

    # Strategy
    ssgan = GANStrategy(data_provider=data_provider,
                        data_sampling_config=sampling_config,
                        d_trainer=d_trainer,
                        g_trainer=g_trainer,
                        n_epochs=args.n_epochs,
                        callbacks=st_callbacks,
                        device=args.device)

    ssgan.run()
