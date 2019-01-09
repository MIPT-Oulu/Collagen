import torch

from collagen.data import ItemLoader, DataProvider
from collagen.data import FoldSplit
from collagen.strategies import Strategy
from collagen.metrics import RunningAverageMeter, AccuracyMeter
from ex_utils import get_mnist, init_mnist_transforms, init_args
from ex_utils import SimpleConvNet


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


if __name__ == "__main__":
    input_args = init_args()

    train_ds, classes = get_mnist(data_folder=input_args.save_mnist, train=True)
    test_ds, _ = get_mnist(data_folder=input_args.save_mnist, train=False)

    model = SimpleConvNet(bw=input_args.bw, drop=input_args.dropout, n_cls=len(classes))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=input_args.lr, weight_decay=input_args.wd)

    if not torch.cuda.is_available() and input_args.device == "cuda":
        raise ValueError("CUDA not found")

    args = {
        "splitter": {"n_folds": 2, "target_col": "target"},
        "itemloader": {"batch_size": input_args.bs, "num_workers": input_args.num_workers, "drop_last": False},
        "train": {"n_epochs": input_args.n_epochs},
        "data": {"data_col": "img", "target_col": "target"}
    }

    train_cbs = (RunningAverageMeter(), )
    val_cbs = (AccuracyMeter(), )

    for fold_id, (df_train, df_val) in enumerate(FoldSplit(train_ds, **args["splitter"])):
        item_loaders = dict()

        for stage, df in zip(['train', 'eval'], [df_train, df_val]):
            item_loaders[f'{fold_id}_{stage}'] = ItemLoader(meta_data=df,
                                                            transform=init_mnist_transforms()[0],
                                                            parse_item_cb=parse_item_mnist,
                                                            **args["itemloader"])

        data_provider = DataProvider(item_loaders)

        strategy = Strategy(data_provider=data_provider,
                            train_loader_names=f'{fold_id}_train',
                            val_loader_names=f'{fold_id}_eval',
                            data_key="img",
                            target_key="target",
                            loss=criterion,
                            model=model,
                            n_epochs=args["train"]["n_epochs"],
                            optimizer=optimizer,
                            train_callbacks=train_cbs,
                            val_callbacks=val_cbs,
                            device=input_args.device)

        strategy.run()

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=init_mnist_transforms()[1], parse_item_cb=parse_item_mnist,
                                      batch_size=input_args.bs, num_workers=input_args.num_workers,
                                      drop_last=False)


