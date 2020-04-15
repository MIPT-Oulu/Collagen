import torch
import numpy as np
import yaml
import random
from tensorboardX import SummaryWriter

from collagen import *

from examples.cnn.utils import init_mnist_transforms, init_args
from examples.cnn.utils import SimpleConvNet

device = auto_detect_device()


def parse_item_mnist(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {data_key: img, target_key: target}


if __name__ == "__main__":
    args = init_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dataset == 'cifar10':
        train_ds, classes = get_cifar10(data_folder=args.save_data, train=True)
        n_channels = 3
    elif args.dataset == 'mnist':
        train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
        n_channels = 1
    else:
        raise ValueError('Not support dataset {}'.format(args.dataset))

    criterion = torch.nn.CrossEntropyLoss()

    # Tensorboard visualization
    log_dir = args.log_dir
    comment = args.comment
    summary_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)
    with open("strategy.yml", "r") as f:
        strategy_config = yaml.load(f)

    for fold_id, (df_train, df_val) in enumerate(splitter):
        item_loaders = dict()

        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                        transform=init_mnist_transforms()[0],
                                                        parse_item_cb=parse_item_mnist,
                                                        batch_size=args.bs, num_workers=args.num_threads,
                                                        shuffle=True if stage == "train" else False)

        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes), n_channels=n_channels)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        data_provider = DataProvider(item_loaders)

        train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                     AccuracyMeter(prefix="train", name="acc"))

        val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
                   AccuracyMeter(prefix="eval", name="acc"),
                   ScalarMeterLogger(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))

        session = Session(data_provider=data_provider,
                          train_loader_names=tuple(sampling_config['train']['data_provider']['mymodel'].keys()),
                          val_loader_names=tuple(sampling_config['eval']['data_provider']['mymodel'].keys()),
                          module=model, loss=criterion, optimizer=optimizer,
                          train_callbacks=train_cbs,
                          val_callbacks=val_cbs)

        strategy = Strategy(data_provider=data_provider,
                            data_sampling_config=sampling_config,
                            strategy_config=strategy_config,
                            sessions=session,
                            n_epochs=args.n_epochs,
                            device=device)

        strategy.run()
