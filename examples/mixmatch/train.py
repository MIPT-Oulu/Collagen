from torch import optim
from tensorboardX import SummaryWriter
import yaml

from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.strategies import Strategy
from collagen.metrics import RunningAverageMeter, AccuracyMeter, KappaMeter
from collagen.data.utils import get_mnist, get_cifar10
from collagen.logging import MeterLogging
from collagen.callbacks.visualizer import ConfusionMatrixVisualizer

from examples.mixmatch.utils import init_args, parse_item, init_transforms
from examples.mixmatch.utils import cond_accuracy_meter, parse_output, parse_target, parse_output_cls, parse_target_cls
from examples.mixmatch.losses import MixMatchModelLoss
from examples.mixmatch.data_provider import mixmatch_data_provider
from examples.mixmatch.networks import Wide_ResNet

device = auto_detect_device()

if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "MixMatch"

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
    model = Wide_ResNet(depth=args.n_depths, widen_factor=args.w_factor, dropout_rate=args.dropout_rate,
                        num_classes=args.n_classes).to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))
    crit = MixMatchModelLoss(alpha=75.0).to(device)

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    data_provider = mixmatch_data_provider(model=model, labeled_meta_data=train_labeled_data, parse_item=parse_item,
                                           unlabeled_meta_data=train_unlabeled_data, bs=args.bs,
                                           augmentation=init_transforms(nc=n_channels)[2], n_augmentations=2,
                                           num_threads=args.num_threads, val_labeled_data=val_labeled_data,
                                           transforms=init_transforms(nc=n_channels))

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Callbacks
    callbacks_train = (RunningAverageMeter(prefix='train', name='loss_x'),
                       RunningAverageMeter(prefix='train', name='loss_u'),
                       MeterLogging(writer=summary_writer),
                       AccuracyMeter(prefix="train", name="acc", parse_target=parse_target, parse_output=parse_output,
                                     cond=cond_accuracy_meter),
                       KappaMeter(prefix='train', name='kappa', parse_target=parse_target_cls,
                                  parse_output=parse_output_cls,
                                  cond=cond_accuracy_meter),
                       ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                 parse_target=parse_target_cls, parse_output=parse_output_cls,
                                                 labels=[str(i) for i in range(10)], tag="train/confusion_matrix"))

    callbacks_eval = (RunningAverageMeter(prefix='eval', name='loss_x'),
                      RunningAverageMeter(prefix='eval', name='loss_u'),
                      AccuracyMeter(prefix="eval", name="acc", parse_target=parse_target, parse_output=parse_output,
                                    cond=cond_accuracy_meter),
                      MeterLogging(writer=summary_writer),
                      KappaMeter(prefix='eval', name='kappa', parse_target=parse_target_cls,
                                 parse_output=parse_output_cls,
                                 cond=cond_accuracy_meter),
                      ConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                parse_target=parse_target_cls, parse_output=parse_output_cls,
                                                labels=[str(i) for i in range(10)], tag="eval/confusion_matrix"))

    st_callbacks = MeterLogging(writer=summary_writer)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    mixmatch = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"].keys()),
                        data_sampling_config=sampling_config,
                        loss=crit,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optim,
                        train_callbacks=callbacks_train,
                        val_callbacks=callbacks_eval,
                        device=device)

    mixmatch.run()
