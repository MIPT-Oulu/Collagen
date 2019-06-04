import yaml
from torch import optim
from tensorboardX import SummaryWriter

from collagen.core import Trainer
from collagen.data import SSFoldSplit
from collagen.data.utils import get_cifar10, get_mnist
from collagen.data.data_provider import mt_data_provider
from collagen.core.utils import auto_detect_device
from collagen.strategies import DualModelStrategy
from collagen.logging import MeterLogging
from collagen.losses.ssl import MTLoss
from collagen.callbacks.visualizer import ProgressbarVisualizer
from collagen.metrics import RunningAverageMeter, AccuracyMeter, KappaMeter
from collagen.callbacks.dualmodel import UpdateEMA

from examples.mean_teacher.utils import init_args, parse_item, init_transforms, parse_target, parse_class
from examples.mean_teacher.utils import SSConfusionMatrixVisualizer, cond_accuracy_meter
from examples.mean_teacher.networks import Model01

device = auto_detect_device()

if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "MT"

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
                           labeled_train_size_per_class=400, unlabeled_train_size_per_class=5000,
                           equal_target=True, equal_unlabeled_target=True, shuffle=True, unlabeled_target_col='target')

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    summary_writer = SummaryWriter(log_dir=args.log_dir, comment=comment)

    # Initializing Discriminator
    te_network = Model01(nc=n_channels, ndf=args.n_features).to(device)
    te_optim = optim.Adam(te_network.group_parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    te_crit = MTLoss(alpha_cls=1.0, alpha_st_cons=1.0, alpha_aug_cons=.1).to(device)

    # Initializing Generator
    st_network = Model01(nc=n_channels, ndf=args.n_features).to(device)
    st_optim = optim.Adam(st_network.group_parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    st_crit = MTLoss(alpha_cls=1.0, alpha_st_cons=1.0, alpha_aug_cons=.1).to(device)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    # Initializing the data provider
    data_provider = mt_data_provider(st_model=st_network, te_model=te_network, train_labeled_data=train_labeled_data,
                                     val_labeled_data=val_labeled_data, train_unlabeled_data=train_unlabeled_data,
                                     val_unlabeled_data=val_unlabeled_data, transforms=init_transforms(nc=n_channels),
                                     parse_item=parse_item, bs=args.bs, num_threads=args.num_threads,
                                     output_type='logits')
    # Setting up the callbacks
    stra_cbs = (MeterLogging(writer=summary_writer), ProgressbarVisualizer())

    # Trainers
    st_train_cbs = (RunningAverageMeter(prefix='train/S', name='loss_cls'),
                    RunningAverageMeter(prefix='train/S', name='loss_s_t_cons'),
                    RunningAverageMeter(prefix='train/S', name='loss_aug_cons'),
                    AccuracyMeter(prefix="train/S", name="acc", parse_target=parse_target, cond=cond_accuracy_meter),
                    KappaMeter(prefix='train/S', name='kappa', parse_target=parse_class, parse_output=parse_class,
                               cond=cond_accuracy_meter),
                    SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter, parse_class=parse_class,
                                                labels=[str(i) for i in range(10)], tag="train/S/confusion_matrix"))

    st_eval_cbs = (RunningAverageMeter(prefix='eval/S', name='loss_cls'),
                   RunningAverageMeter(prefix='eval/S', name='loss_s_t_cons'),
                   RunningAverageMeter(prefix='eval/S', name='loss_aug_cons'),
                   AccuracyMeter(prefix="eval/S", name="acc", parse_target=parse_target, cond=cond_accuracy_meter),
                   KappaMeter(prefix='eval/S', name='kappa', parse_target=parse_class, parse_output=parse_class,
                              cond=cond_accuracy_meter),
                   SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter, parse_class=parse_class,
                                               labels=[str(i) for i in range(10)], tag="eval/S/confusion_matrix"))

    te_train_cbs = (UpdateEMA(st_model=st_network, te_model=te_network),)

    te_eval_cbs = (RunningAverageMeter(prefix='eval/T', name='loss_cls'),
                   AccuracyMeter(prefix="eval/T", name="acc", parse_target=parse_target, cond=cond_accuracy_meter),
                   KappaMeter(prefix='eval/T', name='kappa', parse_target=parse_class, parse_output=parse_class,
                              cond=cond_accuracy_meter),
                   SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter, parse_class=parse_class,
                                               labels=[str(i) for i in range(10)], tag="eval/T/confusion_matrix"))

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

    # Strategy
    mt_strategy = DualModelStrategy(data_provider=data_provider, data_sampling_config=sampling_config, model_names=("S", "T"),
                                    m0_trainer=st_trainer, m1_trainer=te_trainer, n_epochs=args.n_epochs,
                                    n_training_batches=args.n_training_batches, callbacks=stra_cbs, device=args.device)

    mt_strategy.run()
