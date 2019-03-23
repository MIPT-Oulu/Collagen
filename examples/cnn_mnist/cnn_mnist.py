import torch
import numpy as np
import yaml

from collagen.data import ItemLoader, DataProvider
from collagen.data import FoldSplit
from collagen.core.utils import auto_detect_device
from collagen.strategies import Strategy
from collagen.metrics import RunningAverageMeter, AccuracyMeter
from collagen.callbacks import ProgressbarVisualizer
from collagen.savers import ModelSaver
import random
from collagen.logging import MeterLogging
from tensorboardX import SummaryWriter
from examples.cnn_mnist.ex_utils import get_mnist, init_mnist_transforms, init_args
from examples.cnn_mnist.ex_utils import SimpleConvNet

device = auto_detect_device()


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


if __name__ == "__main__":
    args = init_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_ds, classes = get_mnist(data_folder=args.data_dir, train=True)
    test_ds, _ = get_mnist(data_folder=args.data_dir, train=False)

    criterion = torch.nn.CrossEntropyLoss()

    # Tensorboard visualization
    log_dir = args.log_dir
    comment = args.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    kfold_train_losses = []
    kfold_val_losses = []
    kfold_val_accuracies = []

    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    for fold_id, (df_train, df_val) in enumerate(splitter):
        item_loaders = dict()

        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'{fold_id}_{stage}'] = ItemLoader(meta_data=df,
                                                            transform=init_mnist_transforms()[0],
                                                            parse_item_cb=parse_item_mnist,
                                                            batch_size=args.bs, num_workers=args.num_threads,
                                                            shuffle=True if stage == "train" else False)

        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        data_provider = DataProvider(item_loaders)

        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     ProgressbarVisualizer(update_freq=1))

        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   AccuracyMeter(prefix="eval", name="acc"),
                   ProgressbarVisualizer(update_freq=1),
                   MeterLogging(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))

        strategy = Strategy(data_provider=data_provider,
                            train_loader_names=f'{fold_id}_train',
                            val_loader_names=f'{fold_id}_eval',
                            data_key="img",
                            target_key="target",
                            loss=criterion,
                            model=model,
                            n_epochs=args.n_epochs,
                            optimizer=optimizer,
                            train_callbacks=train_cbs,
                            val_callbacks=val_cbs,
                            device=args.device)

        strategy.run()
        kfold_train_losses.append(train_cbs[0].current())
        kfold_val_losses.append(val_cbs[0].current())
        kfold_val_accuracies.append(val_cbs[1].current())

    print("k-fold training loss: {}".format(np.asarray(kfold_train_losses).mean()))
    print("k-fold validation loss: {}".format(np.asarray(kfold_val_losses).mean()))
    print("k-fold validation accuracy: {}".format(np.asarray(kfold_val_accuracies).mean()))

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=init_mnist_transforms()[1], parse_item_cb=parse_item_mnist,
                                      batch_size=args.bs, num_workers=args.num_workers,
                                      drop_last=False)
