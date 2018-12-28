import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm

from collagen.data import DataProvider, ItemLoader
from collagen.data.utils import FoldSplit
from collagen.core import Session, TrainValStrategy


from ex_utils import get_mnist, init_mnist_transforms, init_args, SimpleConvNet


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


if __name__ == "__main__":
    args = init_args()

    train_ds, classes = get_mnist(data_folder=args.save_mnist, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_mnist, train=False)

    for fold_id, (x_train, x_val) in enumerate(FoldSplit(train_ds, random_state=args.seed)):
        item_loaders = dict()

        kwargs = {'batch_size': args.bs, 'num_workers': args.num_threads}
        for stage, ds, trf in zip(['train', 'eval'], [x_train, x_val], init_mnist_transforms()):
            item_loaders[f'{fold_id}_{stage}'] = ItemLoader(meta_data=ds,
                                                            transform=trf, parse_item_cb=parse_item_mnist,
                                                            **kwargs)

        data_provider = DataProvider(item_loaders)

        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes)).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        se = Session(module=model, optimizer=optimizer, loss=criterion)

        st = TrainValStrategy(data_provider, f'{fold_id}_train', f'{fold_id}_eval', se)

        for epoch in range(args.n_epochs):
            for stage in ['train', 'eval']:
                n_batches = len(item_loaders[f'{fold_id}_{stage}'])
                for batch_i in tqdm(range(n_batches), desc=f'Fold [{fold_id}] | Epoch [{epoch}] | {stage}::'):
                    data_provider.sample(**{f'{fold_id}_{stage}': 1})
                    getattr(st, stage)(data_key='img', target_key='target')

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=init_mnist_transforms()[1], parse_item_cb=parse_item_mnist,
                                      batch_size=args.val_bs, num_workers=args.num_threads,
                                      drop_last=False)


