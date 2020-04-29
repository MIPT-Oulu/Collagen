import os
import torch
import hydra
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

# You can be "straightforward" as below
# from collagen import *
# Or just be a nerd
from collagen.core import Session
from collagen.strategies import Strategy
from collagen.core.utils import auto_detect_device
from collagen.data import get_cifar10, DataProvider, ItemLoader
from collagen.callbacks import RunningAverageMeter, AccuracyMeter, ScalarMeterLogger, ModelSaver, \
    CosineAnnealingWarmRestartsWithWarmup

from examples.cnn_cifar10.utils import my_transforms, parse_item
from examples.cnn_cifar10.model import ResNet

device = auto_detect_device()


@hydra.main(config_path='configs/config.yaml')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_dir = os.path.join(os.environ['PWD'], cfg.data_dir)

    train_ds, classes = get_cifar10(data_folder=data_dir, train=True)
    eval_ds, _ = get_cifar10(data_folder=data_dir, train=False)
    n_channels = 3

    criterion = torch.nn.CrossEntropyLoss()

    model = ResNet(in_channels=n_channels, n_features=64, drop_rate=0.3).to(device).half()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd,
                                nesterov=True)

    # Tensorboard visualization
    log_dir = cfg.log_dir
    comment = cfg.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    item_loaders = dict()
    for stage, df in zip(['train', 'eval'], [train_ds, eval_ds]):
        item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=df,
                                                     transform=my_transforms()[stage],
                                                     parse_item_cb=parse_item,
                                                     batch_size=cfg.bs, num_workers=cfg.num_workers,
                                                     shuffle=True if stage == "train" else False)

    data_provider = DataProvider(item_loaders)

    train_cbs = (CosineAnnealingWarmRestartsWithWarmup(optimizer=optimizer, warmup_epochs=(0, 10, 20),
                                                       warmup_lrs=(0, 0.1, 0.01), T_O=5, T_mult=2, eta_min=0),
                 RunningAverageMeter(name="loss"),
                 AccuracyMeter(name="acc"))

    val_cbs = (RunningAverageMeter(name="loss"),
               AccuracyMeter(name="acc"),
               ScalarMeterLogger(writer=summary_writer),
               ModelSaver(metric_names='loss', save_dir=cfg.snapshots, conditions='min', model=model),
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
