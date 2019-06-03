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

from examples.mean_teacher.utils import init_args, parse_item, init_transforms
from examples.mean_teacher.networks import Model01

device = auto_detect_device()

if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "PI_model"

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

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    summary_writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)

    # Initializing Discriminator
    te_network = Model01(nc=1, ndf=args.d_net_features).to(device)
    te_optim = optim.Adam(te_network.group_parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
    te_crit = MTLoss(alpha_cls=0.6, alpha_st_cons=0.2, alpha_aug_cons=0.2).to(device)

    # Initializing Generator
    st_network = Model01(nc=1, ndf=args.d_net_features).to(device)
    st_optim = optim.Adam(st_network.group_parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
    st_crit = MTLoss(alpha_cls=0.6, alpha_st_cons=0.2, alpha_aug_cons=0.2).to(device)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    # Initializing the data provider
    data_provider = mt_data_provider(st_model=st_network, te_model=te_network, train_labeled_data=train_labeled_data,
                                     val_labeled_data=val_labeled_data, train_unlabeled_data=train_unlabeled_data,
                                     val_unlabeled_data=val_unlabeled_data, transforms=init_transforms(nc=n_channels),
                                     parse_item=parse_item, bs=args.bs, num_threads=args.num_threads, output_type='logits')
    # Setting up the callbacks
    stra_callbacks = MeterLogging(writer=summary_writer)

    # Trainers
    st_trainer = Trainer(data_provider=data_provider,
                         train_loader_names=tuple(sampling_config["train"]["data_provider"]["st"].keys()),
                         val_loader_names=tuple(sampling_config["eval"]["data_provider"]["st"].keys()),
                         module=st_network, optimizer=st_optim, loss=st_crit)

    te_trainer = Trainer(data_provider=data_provider,
                         train_loader_names=tuple(sampling_config["train"]["data_provider"]["te"].keys()),
                         val_loader_names=tuple(sampling_config["eval"]["data_provider"]["te"].keys()),
                         module=te_network, optimizer=te_optim, loss=te_crit)

    # Strategy
    dcgan = DualModelStrategy(data_provider=data_provider, data_sampling_config=sampling_config,
                              model_names=("student", "teacher"),
                              m1_trainer=st_trainer, m2_trainer=te_trainer,
                              n_epochs=args.n_epochs,
                              callbacks=stra_callbacks,
                              device=args.device)

    dcgan.run()
