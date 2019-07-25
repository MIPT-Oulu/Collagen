import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from collagen.core import Module
from collagen.core.utils import auto_detect_device
from collagen.data import SSFoldSplit
from collagen.data.data_provider import pimodel_data_provider
from collagen.data.utils.datasets import get_mnist, get_cifar10
from collagen.callbacks.logging import MeterLogging
from collagen.callbacks.metrics import RunningAverageMeter, AccuracyMeter
from collagen.strategies import Strategy
from examples.pi_model.networks import Model01
from examples.pi_model.utils import SSConfusionMatrixVisualizer, cond_accuracy_meter, parse_class
from examples.pi_model.utils import init_args, parse_item, init_transforms, parse_target_accuracy_meter

device = auto_detect_device()


class PiModelLoss(Module):
    def __init__(self, alpha=0.5, cons_mode='mse'):
        super().__init__()
        self.__cons_mode = cons_mode
        if cons_mode == 'mse':
            self.__loss_cons = self.softmax_mse_loss
        elif cons_mode == 'kl':
            self.__loss_cons = self.softmax_kl_loss
        self.__loss_cls = CrossEntropyLoss(reduction='sum')
        self.__alpha = alpha
        self.__losses = {'loss_cls': None, 'loss_cons': None}

        self.__n_minibatches = 2.0

    @staticmethod
    def softmax_kl_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = target_logits.shape[1]
        return F.kl_div(input_log_softmax, target_softmax, reduction='sum') / n_classes

    @staticmethod
    def softmax_mse_loss(input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_classes = input_logits.size()[1]
        return F.mse_loss(input_softmax, target_softmax, reduction='sum') / n_classes

    def forward(self, pred: Tensor, target: Tensor):
        n_minibatch_size = pred.shape[0]
        if target['name'] == 'u':
            aug_logit = target['logits']

            loss_cons = self.__loss_cons(aug_logit, pred)

            self.__losses['loss_cons'] = self.__alpha * loss_cons / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss_cls'] = None
            self.__losses['loss'] = self.__losses['loss_cons']
            _loss = self.__losses['loss']

        elif target['name'] == 'l':
            aug_logit = target['logits']
            target_cls = target['target'].type(torch.int64)

            loss_cls = self.__loss_cls(pred, target_cls)
            loss_cons = self.__loss_cons(aug_logit, pred)
            self.__losses['loss_cons'] = self.__alpha * loss_cons / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss_cls'] = loss_cls / (self.__n_minibatches * n_minibatch_size)
            self.__losses['loss'] = self.__losses['loss_cls'] + self.__losses['loss_cons']
            _loss = self.__losses['loss']
        else:
            raise ValueError("Not support target name {}".format(target['name']))

        return _loss

    def get_loss_by_name(self, name):
        if name in self.__losses:
            return self.__losses[name]
        else:
            return None

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass


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
                       MeterLogging(writer=summary_writer),
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
                      MeterLogging(writer=summary_writer),
                      SSConfusionMatrixVisualizer(writer=summary_writer, cond=cond_accuracy_meter,
                                                  parse_class=parse_class,
                                                  labels=[str(i) for i in range(10)], tag="eval/confusion_matrix"))

    st_callbacks = MeterLogging(writer=summary_writer)

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
                        device=args.device)

    pi_model.run()
