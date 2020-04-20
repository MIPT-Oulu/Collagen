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
from collagen.data import get_mnist, get_cifar10, FoldSplit, DataProvider, ItemLoader
from collagen.callbacks import RunningAverageMeter, AccuracyMeter, ScalarMeterLogger, ModelSaver

from examples.cnn.utils import my_transforms, parse_item, SimpleConvNet

device = auto_detect_device()


@hydra.main(config_path='configs/config.yaml')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_dir = os.path.join(os.environ['PWD'], cfg.data_dir)

    if cfg.dataset == 'cifar10':
        train_ds, classes = get_cifar10(data_folder=data_dir, train=True)
        n_channels = 3
    elif cfg.dataset == 'mnist':
        train_ds, classes = get_mnist(data_folder=data_dir, train=True)
        n_channels = 1
    else:
        raise ValueError('Not support dataset {}'.format(cfg.dataset))

    criterion = torch.nn.CrossEntropyLoss()

    # Tensorboard visualization
    log_dir = cfg.log_dir
    comment = cfg.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    for fold_id, (df_train, df_val) in enumerate(splitter):
        item_loaders = dict()

        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'loader_{stage}'] = ItemLoader(meta_data=df,
                                                         transform=my_transforms()[stage],
                                                         parse_item_cb=parse_item,
                                                         batch_size=cfg.bs, num_workers=cfg.num_threads,
                                                         shuffle=True if stage == "train" else False)

        model = SimpleConvNet(bw=cfg.bw, drop=cfg.dropout, n_cls=len(classes), n_channels=n_channels)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        data_provider = DataProvider(item_loaders)

        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     AccuracyMeter(prefix="train", name="acc"))

        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   AccuracyMeter(prefix="eval", name="acc"),
                   ScalarMeterLogger(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=cfg.snapshots, conditions='min', model=model))

        session = Session(data_provider=data_provider,
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
