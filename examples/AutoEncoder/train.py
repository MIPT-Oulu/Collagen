# Some functions are copied and/or modified from cnn examples
import torch
import numpy as np
import yaml

from collagen.data import ItemLoader, DataProvider
from collagen.data import FoldSplit
from collagen.core.utils import auto_detect_device

from collagen.strategies import Strategy
from collagen.callbacks import RunningAverageMeter

from collagen.callbacks import ModelSaver
import random
from collagen.callbacks import ScalarMeterLogger
from tensorboardX import SummaryWriter
from examples.AutoEncoder.utils import init_mnist_transforms, init_args, get_mnist32x32
from examples.AutoEncoder.models import AutoEncoder

device = auto_detect_device()


def parse_item_mnist(root, entry, trf, data_key, target_key):
    img, target = trf((entry[data_key], entry[target_key]))
    return {data_key: img, target_key: target}


if __name__ == "__main__":
    args = init_args()
    device = auto_detect_device()
    summary_writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    kfold_train_losses = []
    kfold_val_losses = []
    kfold_val_accuracies = []

    # Initializing the data provider

    train_ds, classes = get_mnist32x32(data_folder=args.save_data, train=True)
    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")

    for fold_id, (df_train, df_val) in enumerate(splitter):
        item_loaders = dict()
        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                        transform=init_mnist_transforms()[0],
                                                        parse_item_cb=parse_item_mnist,
                                                        batch_size=args.bs,
                                                        num_workers=args.num_threads,
                                                        shuffle=True if stage == 'train' else False)
        # Initialize AutoEncoder
        model = AutoEncoder().to(device)
        optimizer = torch.optim.Adam(model.group_parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        criterion = torch.nn.BCELoss().to(device)
        data_provider = DataProvider(item_loaders)

        train_cbs = (RunningAverageMeter(prefix='train', name='loss'))
        val_cbs = (RunningAverageMeter(prefix='eval', name='loss'),
                   ScalarMeterLogger(writer=summary_writer),
                   ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))

        strategy = Strategy(data_provider=data_provider,
                            train_loader_names=tuple(sampling_config['train']['data_provider'].keys()),
                            val_loader_names=tuple(sampling_config['eval']['data_provider'].keys()),
                            data_sampling_config=sampling_config,
                            loss=criterion,
                            model=model,
                            n_epochs=args.n_epochs,
                            optimizer=optimizer,
                            train_callbacks=train_cbs,
                            val_callbacks=val_cbs,
                            device=args.device)
        strategy.run()
        kfold_train_losses.append(train_cbs.current())
        kfold_val_losses.append(val_cbs[0].current())
    print("k-fold training loss: {}".format(np.asarray(kfold_train_losses).mean()))
    print("k-fold validation loss: {}".format(np.asarray(kfold_val_losses).mean()))
