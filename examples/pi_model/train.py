import yaml
from tensorboardX import SummaryWriter
from torch import optim

from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.data.data_provider import pimodel_data_provider
from collagen.data.utils.datasets import get_mnist, get_cifar10
from collagen.callbacks import RunningAverageMeter, AccuracyMeter, ScalarMeterLogger
from collagen.strategies import Strategy
from examples.pi_model.losses import PiModelLoss
from examples.pi_model.networks import Model01
from examples.pi_model.utils import SSConfusionMatrixVisualizer, cond_accuracy_meter, parse_class
from examples.pi_model.utils import init_args, parse_item, init_transforms, parse_target_accuracy_meter

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
        raise ValueError('Not supported dataset {}'.format(args.dataset))

    n_folds = 5
    splitter = SSFoldSplit(train_ds, n_ss_folds=3, n_folds=n_folds, target_col="target", random_state=args.seed,
                           labeled_train_size_per_class=400, unlabeled_train_size_per_class=2000,
                           equal_target=True, equal_unlabeled_target=True, shuffle=True, unlabeled_target_col='target')

    # Initializing Discriminator-like net (similar to dcgan example)
    model = Model01(nc=n_channels, ndf=args.n_features, drop_rate=0.5).to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))
    crit = PiModelLoss(alpha=10.0).to(device)

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)
    t_tr_l = train_labeled_data['target']
    t_va_l = val_labeled_data['target']
    data_provider = pimodel_data_provider(model=model, train_labeled_data=train_labeled_data,
                                          train_unlabeled_data=train_unlabeled_data,
                                          val_labeled_data=val_labeled_data, val_unlabeled_data=val_unlabeled_data,
                                          transforms=init_transforms(nc=n_channels), parse_item=parse_item, bs=args.bs,
                                          num_threads=args.num_threads)

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Callbacks
    callbacks_train = (RunningAverageMeter(prefix='train', name='loss_cls'),
                       RunningAverageMeter(prefix='train', name='loss_cons'),
                       ScalarMeterLogger(writer=summary_writer),
                       AccuracyMeter(prefix="train", name="acc", parse_target=parse_target_accuracy_meter,
                                     cond=cond_accuracy_meter),
                       SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                   parse_class=parse_class,
                                                   labels=[str(i) for i in range(10)], tag="train/confusion_matrix")
                       )

    callbacks_eval = (RunningAverageMeter(prefix='eval', name='loss_cls'),
                      RunningAverageMeter(prefix='eval', name='loss_cons'),
                      AccuracyMeter(prefix="eval", name="acc", parse_target=parse_target_accuracy_meter,
                                    cond=cond_accuracy_meter),
                      ScalarMeterLogger(writer=summary_writer),
                      SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                  parse_class=parse_class,
                                                  labels=[str(i) for i in range(10)], tag="eval/confusion_matrix"))

    st_callbacks = ScalarMeterLogger(writer=summary_writer)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    pi_model = Strategy(data_provider=data_provider,
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

    pi_model.run()
