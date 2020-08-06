import torch
import numpy as np
import yaml

from collagen.data import FoldSplit
from collagen.core.utils import auto_detect_device

from collagen.strategies import Strategy
from collagen.data.utils.datasets import get_mnist

import random
from tensorboardX import SummaryWriter

from examples.autoencoder.utils import init_args, init_data_provider, init_callbacks
from examples.autoencoder.models import AutoEncoder

from examples.autoencoder.sampler import VisualizationSampler

device = auto_detect_device()

if __name__ == "__main__":
    # parse the arguments
    args = init_args()
    # detect device
    device = auto_detect_device()
    # summary writer for tensorboard
    summary_writer = SummaryWriter(log_dir=args.log_dir, comment=args.comment)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open("settings.yml", "r") as f:
        sampling_config = yaml.load(f)

    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    splitter = FoldSplit(train_ds, n_folds=5, target_col="target")
    test_ds, classes = get_mnist(data_folder=args.save_data, train=False)

    # Get data of the first fold
    df_train, df_val = next(splitter)

    item_loaders = dict()
    data_provider = init_data_provider(args, df_train, df_val, item_loaders, test_ds)

    model = AutoEncoder(16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = torch.nn.MSELoss().to(device)
    viz_sampler = VisualizationSampler(viz_loader=item_loaders['mnist_viz'],
                                       device=device, bs=args.bs, ae=model)
    callbacks = init_callbacks(args, summary_writer, model, viz_sampler)

    strategy = Strategy(data_provider=data_provider,
                        train_loader_names=tuple(sampling_config['train']['data_provider'].keys()),
                        val_loader_names=tuple(sampling_config['eval']['data_provider'].keys()),
                        data_sampling_config=sampling_config,
                        loss=criterion,
                        model=model,
                        n_epochs=args.n_epochs,
                        optimizer=optimizer,
                        train_callbacks=callbacks['train'],
                        val_callbacks=callbacks['eval'],
                        device=device)
    strategy.run()
