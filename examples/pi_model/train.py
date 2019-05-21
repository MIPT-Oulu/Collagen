from torch.nn import MSELoss, CrossEntropyLoss
from torch import optim, Tensor
import torch
from tensorboardX import SummaryWriter
import yaml

from collagen.core import Module
from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.data.data_provider import pimodel_data_provider
from collagen.strategies import Strategy
from collagen.metrics import RunningAverageMeter, AccuracyMeter
from collagen.data.utils import get_mnist, get_cifar10
from collagen.logging import MeterLogging

from examples.pi_model.utils import init_args, parse_item, init_transforms, parse_target_accuracy_meter
from examples.pi_model.utils import SSConfusionMatrixVisualizer, cond_accuracy_meter
from examples.pi_model.networks import Model01

device = auto_detect_device()


class PiModelLoss(Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.__loss_mse = MSELoss()
        self.__loss_cls = CrossEntropyLoss()
        self.__alpha = alpha
        self.__losses = {'loss_cls': None, 'loss_cons': None}

    def forward(self, pred: Tensor, target: Tensor):
        if isinstance(target, tuple) and len(target) == 2:
            target_cls = target[0].type(torch.int64)
            features = target[1].view(target[1].shape[0], -1)
            loss_cls = self.__loss_cls(pred, target_cls)
            if features.shape[0] != 2:
                raise ValueError("Num of features must be 2, but found {}".format(features.shape[0]))
            loss_cons = self.__loss_mse(features[0, :], features[1, :])
            _loss = self.__alpha*loss_cls + (1 - self.__alpha)*loss_cons
            self.__losses['loss_cons'] = loss_cons
            self.__losses['loss_cls'] = loss_cls
            self.__losses['loss'] = _loss

        elif (isinstance(target, tuple) and len(target) == 1) or (isinstance(target, Tensor) and len(target.shape) == 1):
            target_cls = target.type(torch.int64)
            _loss = self.__loss_cls(pred, target_cls)
            self.__losses['loss_cons'] = None
            self.__losses['loss_cls'] = _loss
            self.__losses['loss'] = _loss
        else:
            raise ValueError("Target length is 1 or 2, but found {}".format(len(target)))

        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None


if __name__ == "__main__":
    args = init_args()
    log_dir = args.log_dir
    comment = "PI_model"

    # Data provider
    dataset_name = 'cifar10'

    if dataset_name == 'cifar10':
        train_ds, classes = get_cifar10(data_folder=args.save_data, train=True)
        n_channels = 3
    elif dataset_name == 'mnist':
        train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
        n_channels = 1
    else:
        raise ValueError('Not support dataset {}'.format(dataset_name))

    n_folds = 5
    splitter = SSFoldSplit(train_ds, n_ss_folds=3, n_folds=n_folds, target_col="target", random_state=args.seed,
                           labeled_train_size_per_class=400, unlabeled_train_size_per_class=2000,
                           equal_target=True, equal_unlabeled_target=True, shuffle=True, unlabeled_target_col='target')

    # Initializing Discriminator
    model = Model01(nc=n_channels, ndf=args.n_features).to(device)
    optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(args.beta1, 0.999))
    crit = PiModelLoss(alpha=0.5).to(device)

    train_labeled_data, val_labeled_data, train_unlabeled_data, val_unlabeled_data = next(splitter)

    data_provider = pimodel_data_provider(model=model, train_labeled_data=train_labeled_data, train_unlabeled_data=train_unlabeled_data,
                                          val_labeled_data=val_labeled_data, val_unlabeled_data=val_unlabeled_data,
                                          transforms=init_transforms(nc=n_channels), parse_item=parse_item, bs=args.bs, num_threads=args.num_threads)

    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)
    # Callbacks
    callbacks_train = (RunningAverageMeter(prefix='train', name='loss_cls'),
                       RunningAverageMeter(prefix='train', name='loss_cons'),
                       MeterLogging(writer=summary_writer),
                       AccuracyMeter(prefix="train", name="acc", parse_target=parse_target_accuracy_meter, cond=cond_accuracy_meter))


    callbacks_eval = (RunningAverageMeter(prefix='train', name='loss_cls'),
                      RunningAverageMeter(prefix='train', name='loss_cons'),
                      AccuracyMeter(prefix="eval", name="acc", parse_target=parse_target_accuracy_meter, cond=cond_accuracy_meter),
                      MeterLogging(writer=summary_writer),
                      SSConfusionMatrixVisualizer(writer=summary_writer,
                                                  labels=[str(i) for i in range(10)],
                                                  tag="eval/confusion_matrix"))

    st_callbacks = MeterLogging(writer=summary_writer)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    pi_model = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config["train"]["data_provider"].keys()),
                        val_loader_names=tuple(sampling_config["eval"]["data_provider"].keys()),
                        data_key=("data", "data"),
                        target_key=("target", "features"),
                        data_sampling_config=sampling_config,
                        loss=crit,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optim,
                        train_callbacks=callbacks_train,
                        val_callbacks=callbacks_eval,
                        device=args.device)

    pi_model.run()
