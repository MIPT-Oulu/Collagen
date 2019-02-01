from torch.nn import BCELoss, CrossEntropyLoss
from torch import optim, Tensor, argmax
from tensorboardX import SummaryWriter
import pandas as pd
import yaml

from collagen.core import Module, Session, Trainer
from collagen.callbacks import OnGeneratorBatchFreezer, OnDiscriminatorBatchFreezer, OnSamplingFreezer
from collagen.callbacks import ProgressbarVisualizer, TensorboardSynthesisVisualizer, ClipGradCallback
from collagen.data import DataProvider, ItemLoader, SSGANFakeSampler, SSFoldSplit, GaussianNoiseSampler
from collagen.strategies import SSGANStrategy
from collagen.metrics import RunningAverageMeter, SSAccuracyMeter, SSValidityMeter
from collagen.data.utils import get_mnist, auto_detect_device
from collagen.logging import MeterLogging

from ex_utils import init_args, parse_item_mnist_ssgan, init_mnist_transforms
from ex_utils import Discriminator, Generator
from ex_utils import load_meta_with_imgs, load_oai_most_datasets


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

            _loss =  self.__alpha * loss_valid + (1 - self.__alpha) * loss_cls
        else:
            target_valid = target
            _loss = self.__loss_valid(pred_valid, target_valid)

        # target_cls = target_cls.type(torch.int64)
        return  _loss


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


if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "ssgan"

    # oai_meta, most_meta, oai_ppl, most_ppl = read_oa_metadata(img_dir=args.img_dir, oai_src_dir=args.oai_meta, most_src_dir=args.most_meta, meta_dir=args.save_meta)
    meta_dict = load_oai_most_datasets(root='./data', save_dir='./Metadata', force_reload=True)
    df_meta = load_meta_with_imgs(meta_dict["oai_most"]["all"], img_dir='./data/MOST_OAI_00_0_2/', saved_patch_dir='./data/MOST_OAI_00_0_2_cropped', force_rewrite=False)
    df_meta.to_csv('./Metadata/oai_most_img_patches.csv', index=None, sep='|')
    
    # Data provider
    n_folds = 10
    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    splitter = SSFoldSplit(train_ds, n_folds=n_folds, target_col="target", unlabeled_size=0.5, val_size=0.4,
                           shuffle=True)

    num_folds_dict = dict()
    num_folds_dict["real_labeled_train"] = 1
    num_folds_dict["real_unlabeled_train"] = 5
    num_folds_dict["real_labeled_val"] = 4

    n_requested_folds = num_folds_dict["real_labeled_val"] + num_folds_dict["real_labeled_train"] + num_folds_dict[
        "real_unlabeled_train"]
    if n_requested_folds > n_folds:
        raise ValueError(
            "Number of requested folds must be less or equal to input number of folds, but found {} > {}".
                format(n_requested_folds, n_folds))

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
    train_labeled_data, train_unlabeled_data, val_labeled_data = next(splitter)

    item_loaders["real_labeled_train"] = ItemLoader(meta_data=train_labeled_data,
                                                    transform=init_mnist_transforms()[1],
                                                    parse_item_cb=parse_item_mnist_ssgan,
                                                    batch_size=args.bs, num_workers=args.num_threads)

    item_loaders["real_unlabeled_train"] = ItemLoader(meta_data=train_unlabeled_data,
                                                      transform=init_mnist_transforms()[1],
                                                      parse_item_cb=parse_item_mnist_ssgan,
                                                      batch_size=args.bs, num_workers=args.num_threads)

    item_loaders["real_labeled_val"] = ItemLoader(meta_data=val_labeled_data,
                                                  transform=init_mnist_transforms()[1],
                                                  parse_item_cb=parse_item_mnist_ssgan,
                                                  batch_size=args.bs, num_workers=args.num_threads)

    item_loaders['fake_unlabeled_gen'] = SSGANFakeSampler(g_network=g_network,
                                                            batch_size=args.bs,
                                                            latent_size=args.latent_size,
                                                            n_classes=args.n_classes)

    item_loaders['fake_unlabeled_latent'] = SSGANFakeSampler(g_network=g_network,
                                                          batch_size=args.bs,
                                                          latent_size=args.latent_size,
                                                          n_classes=args.n_classes)

    # item_loaders['fake_unlabeled_latent'] = GaussianNoiseSampler(batch_size=args.bs, latent_size=args.latent_size,
    #                                                              n_classes=args.n_classes, device=device)

    # item_loaders['fake_unlabeled_latent'] = SSGANFakeSampler(g_network=g_network,
    #                                                          batch_size=args.bs,
    #                                                          latent_size=args.latent_size,
    #                                                          n_classes=args.n_classes)

    data_provider = DataProvider(item_loaders)

    # Callbacks
    g_callbacks_train = (RunningAverageMeter(prefix="train/G", name="loss"),
                         OnGeneratorBatchFreezer(modules=d_network),
                         ClipGradCallback(g_network, mode="norm", max_norm=0.1, norm_type=2))

    g_callbacks_eval = (RunningAverageMeter(prefix="eval/G", name="loss"),)

    d_callbacks_train = (RunningAverageMeter(prefix="train/D", name="loss"),
                         # BackwardCallback(retain_graph=True),
                         SSValidityMeter(threshold=0.5, sigmoid=False, prefix="train/D", name="ss_valid"),
                         SSAccuracyMeter(prefix="train/D", name="ss_acc"),
                         OnDiscriminatorBatchFreezer(modules=g_network))

    d_callbacks_eval = (RunningAverageMeter(prefix="eval/D", name="loss"),
                        SSValidityMeter(threshold=0.5, sigmoid=False, prefix="eval/D", name="ss_valid"),
                        SSAccuracyMeter(prefix="eval/D", name="ss_acc"))

    st_callbacks = (ProgressbarVisualizer(update_freq=1),
                    MeterLogging(writer=summary_writer),
                    OnSamplingFreezer(modules=(g_network, d_network)),
                    TensorboardSynthesisVisualizer(generator_sampler=item_loaders['fake_unlabeled_gen'],
                                                   writer=summary_writer,
                                                   grid_shape=args.grid_shape))

    d_session = Session(module=d_network,
                        optimizer=d_optim,
                        loss=d_crit)

    g_session = Session(module=g_network,
                        optimizer=g_optim,
                        loss=g_crit)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

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
    ssgan = SSGANStrategy(data_provider=data_provider,
                          data_sampling_config=sampling_config,
                          d_trainer=d_trainer,
                          g_trainer=g_trainer,
                          n_epochs=args.n_epochs,
                          callbacks=st_callbacks,
                          device=args.device)

    ssgan.run()
