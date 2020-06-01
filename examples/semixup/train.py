import os
import torch
import numpy as np
import random
import hydra
from docutils.nodes import option

from torch.utils.tensorboard import SummaryWriter

# You can be "straightforward" as below
# from collagen import *
# Or just be a nerd
from collagen.core import Session
from collagen.strategies import Strategy
from collagen.core.utils import auto_detect_device
from collagen.data import get_cifar10
from collagen.callbacks import RunningAverageMeter, AccuracyMeter, ScalarMeterLogger, ModelSaver, MultiLinearByBatchScheduler, \
    CosineAnnealingWarmRestartsWithWarmup

from examples.semixup.data_provider import semixup_data_provider
from examples.semixup.utils import my_transforms, parse_item, data_rearrange, parse_class, cond_accuracy_meter
from examples.semixup.model import EfficientNet, DawnNet
from examples.semixup.losses import SemixupLoss

device = auto_detect_device()


@hydra.main(config_path='configs/config.yaml')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_dir = os.path.join(os.environ['PWD'], cfg.data_dir)

    train_ds, n_classes = get_cifar10(data_folder=data_dir, train=True)
    eval_ds, _ = get_cifar10(data_folder=data_dir, train=False)
    criterion = SemixupLoss(in_manifold_coef=8.0, in_out_manifold_coef=8.0, ic_coef=16.0)

    # Tensorboard visualization
    log_dir = cfg.log_dir
    comment = cfg.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    model = DawnNet(in_channels=3, n_features=64, drop_rate=cfg.dropout).to(device)
    if cfg.pretrained_model:
        print(f'Loading model: {cfg.pretrained_model}')
        model.load_state_dict(torch.load(cfg.pretrained_model))
    optimizer = torch.optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd,
                                nesterov=True)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    # model = EfficientNet.from_name(cfg.arch_name).to(device)
    # if cfg.pretrained_model is not None and cfg.pretrained_model:
    #     print(f'Loading model: {cfg.pretrained_model}')
    #     model.load_state_dict(torch.load(cfg.pretrained_model))

    data_provider = semixup_data_provider(model=model, alpha=np.random.beta(cfg.alpha, cfg.alpha),
                                          n_classes=n_classes,
                                          train_labeled_data=train_ds, train_unlabeled_data=train_ds,
                                          val_labeled_data=eval_ds, transforms=my_transforms(),
                                          parse_item=parse_item, bs=cfg.bs, num_workers=cfg.num_workers,
                                          augmentation=my_transforms()['transforms'], data_rearrange=data_rearrange)

    train_cbs = (CosineAnnealingWarmRestartsWithWarmup(optimizer=optimizer, warmup_epochs=10, warmup_lrs=(1e-9, 0.1),
                                                       T_O=5, T_mult=1, eta_min=0),
                 # CycleRampUpDownScheduler(optimizer, initial_lr=0, rampup_epochs=15, rampup_lr=0.4,
                 #                          start_cycle_epoch=20,
                 #                          rampdown_epochs=25, cycle_interval=5, cycle_rampdown_epochs=0),
                 # MultiLinearByBatchScheduler(optimizer=optimizer, n_batches=len(data_provider.get_loader_by_name('labeled_train')),
                 #                             # steps=[0, 5, 10, 15, 20, 80], lrs=[0.05, 0.005, 0.01, 0.005, 0.001, 1e-8]),
                 #                             steps=[0, 15, 30, 80], lrs=[0, 0.1, 0.005, 1e-8]),
                 RunningAverageMeter(name="loss_cls"),
                 RunningAverageMeter(name="loss_in_mnf"),
                 RunningAverageMeter(name="loss_inout_mnf"),
                 RunningAverageMeter(name="loss_ic"),
                 AccuracyMeter(name="acc", cond=cond_accuracy_meter, parse_output=parse_class,
                               parse_target=parse_class))

    val_cbs = (RunningAverageMeter(name="loss_cls"),
               AccuracyMeter(name="acc", cond=cond_accuracy_meter, parse_output=parse_class, parse_target=parse_class),
               ScalarMeterLogger(writer=summary_writer),
               ModelSaver(metric_names='loss_cls', save_dir=cfg.snapshots, conditions='min', model=model),
               ModelSaver(metric_names='acc', save_dir=cfg.snapshots, conditions='max', model=model))

    session = dict()
    session['mymodel'] = Session(data_provider=data_provider,
                                      train_loader_names=cfg.sampling.train.data_provider.mymodel.keys(),
                                      val_loader_names=cfg.sampling.eval.data_provider.mymodel.keys(),
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
