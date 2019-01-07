import torch

from collagen.data import ItemLoader
from collagen.data import FoldSplit
from collagen.strategies import Strategy
from collagen.core import Callback

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
        "splitter": {"n_folds": 5, "target_col": "target"},
        "itemloader": {"batch_size": input_args.bs, "num_workers": input_args.num_workers, "drop_last": False},
        "train": {"n_epochs": input_args.n_epochs},
        "data": {"data_col": "img", "target_col": "target"}
    }

    train_cbs = Callback()
    val_cbs = Callback()

    strategy = Strategy(data_frame=train_ds,
                        args=args,
                        splitter=FoldSplit,
                        loss=criterion,
                        model=model,
                        optimizer=optimizer,
                        transform=init_mnist_transforms()[0],
                        parse_item_cb=parse_item_mnist,
                        train_callbacks=train_cbs,
                        val_callbacks=val_cbs,
                        device=input_args.device)

    strategy.run()

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=init_mnist_transforms()[1], parse_item_cb=parse_item_mnist,
                                      batch_size=input_args.bs, num_workers=input_args.num_threads,
                                      drop_last=False)


