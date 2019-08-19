from torch import optim
from tensorboardX import SummaryWriter
import yaml

from collagen.core import Trainer
from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.strategies import DualModelStrategy
from collagen.callbacks import RunningAverageMeter, AccuracyMeter, KappaMeter
from collagen.data.utils.datasets import get_mnist, get_cifar10
from collagen.callbacks import ScalarMeterLogger
from collagen.callbacks import ConfusionMatrixVisualizer, ProgressbarLogger
from collagen.callbacks import SetTeacherTrain, UpdateEMA

from examples.mixmatch.utils import init_args, parse_item, init_transforms
from examples.mixmatch.utils import cond_accuracy_meter, parse_output, parse_target, parse_output_cls, parse_target_cls
from examples.mixmatch.losses import MixMatchModelLoss
from examples.mixmatch.data_provider import mixmatch_ema_data_provider
from examples.mixmatch.networks import Wide_ResNet


if __name__ == "__main__":
    device = auto_detect_device()

    args = init_args()
    log_dir = args.log_dir
    comment = "MixMatchEMA"

    # Data provider
    dataset_name = 'cifar10'

    if args.dataset == 'cifar10':
        train_ds, classes = get_cifar10(data_folder=args.save_data, train=True)
        n_channels = 3
    elif args.dataset == 'mnist':
        train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
        n_channels = 1
    else:
        raise ValueError('Not support dataset {}'.format(args.dataset))

    n_folds = 5
    splitter = SSFoldSplit(train_ds, n_ss_folds=3, n_folds=n_folds, target_col="target", random_state=args.seed,
                           labeled_train_size_per_class=400, unlabeled_train_size_per_class=2000,
                           equal_target=True, equal_unlabeled_target=True, shuffle=True, unlabeled_target_col='target')

    # Initializing Discriminator
    st_network = Wide_ResNet(depth=args.n_depths, widen_factor=args.w_factor, dropout_rate=args.dropout_rate,
                             num_classes=args.n_classes).to(device)
    te_network = Wide_ResNet(depth=args.n_depths, widen_factor=args.w_factor, dropout_rate=args.dropout_rate,
                             num_classes=args.n_classes).to(device)
    st_optim = optim.Adam(st_network.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))
    te_optim = optim.Adam(te_network.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))

    st_crit = MixMatchModelLoss(alpha=75.0).to(device)
    te_crit = MixMatchModelLoss(alpha=75.0).to(device)

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    # Use teacher network here
    data_provider = mixmatch_ema_data_provider(model=te_network, labeled_meta_data=train_labeled_data,
                                               parse_item=parse_item,
                                               unlabeled_meta_data=train_unlabeled_data, bs=args.bs,
                                               augmentation=init_transforms(nc=n_channels)[2], n_augmentations=2,
                                               num_threads=args.num_threads, val_labeled_data=val_labeled_data,
                                               transforms=init_transforms(nc=n_channels))

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Callbacks
    scheme_cbs = (ScalarMeterLogger(writer=summary_writer), ProgressbarLogger())

    st_train_cbs = (RunningAverageMeter(prefix='train/S', name='loss_x'),
                    RunningAverageMeter(prefix='train/S', name='loss_u'),
                    ScalarMeterLogger(writer=summary_writer),
                    AccuracyMeter(prefix="train/S", name="acc", parse_target=parse_target, parse_output=parse_output,
                                  cond=cond_accuracy_meter),
                    KappaMeter(prefix='train/S', name='kappa', parse_target=parse_target_cls,
                               parse_output=parse_output_cls,
                               cond=cond_accuracy_meter),
                    ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                              parse_target=parse_target_cls, parse_output=parse_output_cls,
                                              labels=[str(i) for i in range(10)], tag="train/S/confusion_matrix"))

    st_eval_cbs = (RunningAverageMeter(prefix='eval/S', name='loss_x'),
                   RunningAverageMeter(prefix='eval/S', name='loss_u'),
                   AccuracyMeter(prefix="eval/S", name="acc", parse_target=parse_target, parse_output=parse_output,
                                 cond=cond_accuracy_meter),
                   ScalarMeterLogger(writer=summary_writer),
                   KappaMeter(prefix='eval/S', name='kappa', parse_target=parse_target_cls,
                              parse_output=parse_output_cls,
                              cond=cond_accuracy_meter),
                   ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                             parse_target=parse_target_cls, parse_output=parse_output_cls,
                                             labels=[str(i) for i in range(10)], tag="eval/S/confusion_matrix"))

    te_train_cbs = (SetTeacherTrain(te_network), UpdateEMA(st_model=st_network, te_model=te_network, decay=0.99),)

    te_eval_cbs = (RunningAverageMeter(prefix='eval/T', name='loss_cls'),
                   AccuracyMeter(prefix="eval/T", name="acc", parse_target=parse_target, cond=cond_accuracy_meter),
                   KappaMeter(prefix='eval/T', name='kappa', parse_target=parse_target_cls,
                              parse_output=parse_output_cls, cond=cond_accuracy_meter),
                   ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                             parse_output=parse_output_cls, parse_target=parse_target_cls,
                                             labels=[str(i) for i in range(10)], tag="eval/T/confusion_matrix"))

    st_callbacks = ScalarMeterLogger(writer=summary_writer)

    with open("settings_ema.yml", "r") as f:
        sampling_config = yaml.load(f)

    st_trainer = Trainer(data_provider=data_provider,
                         train_loader_names=tuple(sampling_config["train"]["data_provider"]["S"].keys()),
                         val_loader_names=tuple(sampling_config["eval"]["data_provider"]["S"].keys()),
                         module=st_network, optimizer=st_optim, loss=st_crit,
                         train_callbacks=st_train_cbs, val_callbacks=st_eval_cbs)

    te_trainer = Trainer(data_provider=data_provider,
                         train_loader_names=None,
                         val_loader_names=tuple(sampling_config["eval"]["data_provider"]["T"].keys()),
                         module=te_network, optimizer=te_optim, loss=te_crit,
                         train_callbacks=te_train_cbs, val_callbacks=te_eval_cbs)

    mixmatch = DualModelStrategy(data_provider=data_provider, data_sampling_config=sampling_config,
                                 model_names=("S", "T"),
                                 m0_trainer=st_trainer, m1_trainer=te_trainer, n_epochs=args.n_epochs,
                                 n_training_batches=args.n_training_batches, callbacks=scheme_cbs, device=device)

    mixmatch.run()
