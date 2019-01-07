import torch

from collagen.data import ItemLoader
from collagen.data import FoldSplit
from collagen.strategies import Strategy
import numpy as np

from .ex_utils import get_mnist, init_mnist_transforms, init_args, SimpleConvNet


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


if __name__ == "__main__":
    args = init_args()

    train_ds, classes = get_mnist(data_folder=args.save_mnist, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_mnist, train=False)

    model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes))
    kwargs = {'batch_size': args.bs, 'num_workers': args.num_threads}
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)

    strategy = Strategy(data_frame=train_ds,
                        args=kwargs,
                        splitter=FoldSplit,
                        loss=criterion,
                        model=model,
                        optimizer=optimizer,
                        transform=init_mnist_transforms,
                        parse_item_cb=parse_item_mnist)

    strategy.run()

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=init_mnist_transforms()[1], parse_item_cb=parse_item_mnist,
                                      batch_size=args.bs, num_workers=args.num_threads,
                                      drop_last=False)


