import os
import torch
import numpy as np
import random
import hydra

from torch.utils.tensorboard import SummaryWriter

# You can be "straightforward" as below
# from collagen import *
# Or just be a nerd
from collagen.core import Session
from collagen.strategies import Strategy
from collagen.core.utils import auto_detect_device
from collagen.data import get_cifar10, FoldSplit, SSFoldSplit
from collagen.callbacks import RunningAverageMeter, BalancedAccuracyMeter, AccuracyMeter, ScalarMeterLogger, ModelSaver

from examples.semixup.data_provider import semixup_data_provider
from examples.semixup.utils import my_transforms, parse_item, data_rearrange, parse_class, cond_accuracy_meter
from examples.semixup.model import EfficientNet
from examples.semixup.losses import SemixupLoss

device = auto_detect_device()


@hydra.main(config_path='configs/config.yaml')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_dir = os.path.join(os.environ['PWD'], cfg.data_dir)

    if cfg.dataset == 'cifar10':
        train_ds, n_classes = get_cifar10(data_folder=data_dir, train=True)
        n_channels = 3
    else:
        raise ValueError('Not support dataset {}'.format(cfg.dataset))

    criterion = SemixupLoss(in_manifold_coef=2.0, in_out_manifold_coef=2.0, ic_coef=4.0)

    # Tensorboard visualization
    log_dir = cfg.log_dir
    comment = cfg.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    for fold_id, (df_train, df_val) in enumerate(splitter):
        model = EfficientNet.from_name(cfg.arch_name).to(device)
        data_provider = semixup_data_provider(model=model, alpha=np.random.beta(cfg.alpha, cfg.alpha), n_classes=n_classes,
                                              train_labeled_data=df_train, train_unlabeled_data=df_train,
                                              val_labeled_data=df_val, transforms=my_transforms(),
                                              parse_item=parse_item, bs=cfg.bs, num_workers=cfg.num_workers,
                                              augmentation=my_transforms()['transforms'], data_rearrange=data_rearrange)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     AccuracyMeter(prefix="train", name="acc", cond=cond_accuracy_meter, parse_output=parse_class,
                                   parse_target=parse_class),
                     # BalancedAccuracyMeter(prefix="train", name="bac", cond=cond_accuracy_meter, parse_output=parse_class,
                     #                       parse_target=parse_class)
                     )

        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   AccuracyMeter(prefix="eval", name="acc", cond=cond_accuracy_meter, parse_output=parse_class,
                                 parse_target=parse_class),
                   # BalancedAccuracyMeter(prefix="eval", name="bac", cond=cond_accuracy_meter, parse_output=parse_class,
                   #                       parse_target=parse_class),
                   ScalarMeterLogger(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=cfg.snapshots, conditions='min', model=model))

        session = dict()
        session['efficientnet'] = Session(data_provider=data_provider,
                                     train_loader_names=cfg.sampling.train.data_provider.efficientnet.keys(),
                                     val_loader_names=cfg.sampling.eval.data_provider.efficientnet.keys(),
                                     module=model, loss=criterion, optimizer=optimizer,
                                     train_callbacks=train_cbs,
                                     val_callbacks=val_cbs)

        strategy = Strategy(data_provider=data_provider,
                            data_sampling_config=cfg.sampling,
                            strategy_config=cfg.strategy,
                            sessions=session,
                            n_epochs=cfg.n_epochs,
                            device=device)

        strategy.run()


if __name__ == "__main__":
    main()
