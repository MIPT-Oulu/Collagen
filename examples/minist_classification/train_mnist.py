import torch
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm

from collagen.data import DataProvider, ItemLoader
from collagen.core import Session, TrainValStrategy


from ex_utils import get_mnist, init_mnist_transforms, init_args, SimpleConvNet


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


if __name__ == "__main__":
    args = init_args()

    # Initializing the dataframes
    train_ds, classes = get_mnist(data_folder=args.save_mnist, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_mnist, train=False)

    train_trf, test_trf = init_mnist_transforms()


    for fold_id, (train_idx, val_idx) in enumerate(cv_folds):
        item_loaders = dict()

        item_loaders[f'{fold_id}_train'] = ItemLoader(root='', meta_data=train_ds.iloc[train_idx],
                                                      transform=train_trf, parse_item_cb=parse_item_mnist,
                                                      batch_size=args.bs, num_workers=args.num_threads)

        item_loaders[f'{fold_id}_val'] = ItemLoader(root='', meta_data=train_ds.iloc[val_idx],
                                                    transform=test_trf, parse_item_cb=parse_item_mnist,
                                                    batch_size=args.val_bs, num_workers=args.num_threads,
                                                    drop_last=False)

        data_provider = DataProvider(item_loaders)

        model = SimpleConvNet(bw=args.bw, drop=args.dropout, n_cls=len(classes))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = torch.nn.CrossEntropyLoss()
        se = Session(module=model, optimizer=optimizer, loss=criterion)
        st = TrainValStrategy(data_provider, f'{fold_id}_train', f'{fold_id}_val', se)

        for epoch in range(args.n_epochs):
            for stage in ['train', 'val']:
                for batch_i in tqdm(range(len(item_loaders[f'{fold_id}_{stage}'])), desc=f'{stage}::'):
                    data_provider.sample(**{f'{fold_id}_train': 1})
                    st.train(data_key='img', target_key='target')

    item_loaders = dict()
    item_loaders['test'] = ItemLoader(root='', meta_data=test_ds,
                                      transform=test_trf, parse_item_cb=parse_item_mnist,
                                      batch_size=args.val_bs, num_workers=args.num_threads,
                                      drop_last=False)


