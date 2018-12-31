import torch
import argparse
from tqdm import tqdm
import numpy as np

import solt.core as slc
import solt.transforms as slt

from collagen.data import DataProvider, ItemLoader
from collagen.data.utils import FoldSplit
from collagen.core import Session, TrainValStrategy
from collagen.models import DCGAN
from collagen.data.utils import get_mnist, ApplyTransform, Normalize, Compose
from collagen.data.utils import wrap2solt, unpack_solt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_item_mnist(root, entry, trf):
    img, target = trf((entry.img, entry.target))
    return {'img': img, 'target': target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def init_mnist_transforms():
    train_trf = Compose([
        wrap2solt,
        slc.Stream([
            slt.RandomScale(range_x=(0.9, 1.1), same=False, p=0.5),
            slt.RandomShear(range_x=(-0.05, 0.05), p=0.5),
            slt.RandomRotate(rotation_range=(-5, 5), p=0.5),
            slt.PadTransform(pad_to=34),
            slt.CropTransform(crop_size=32, crop_mode='r')
        ]),
        unpack_solt,
        ApplyTransform(Normalize((0.1307,), (0.3081,)))
    ])

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(Normalize((0.1307,), (0.3081,)))
    ])

    return train_trf, test_trf


if __name__ == "__main__":
    args = init_args()

    train_ds, classes = get_mnist(data_folder=args.save_data, train=True)
    test_ds, _ = get_mnist(data_folder=args.save_data, train=False)

    for fold_id, (x_train, x_val) in enumerate(FoldSplit(train_ds, random_state=args.seed)):
        item_loaders = dict()

        kwargs = {'batch_size': args.bs, 'num_workers': args.num_threads}
        for stage, ds, trf in zip(['train', 'eval'], [x_train, x_val], init_mnist_transforms()):
            item_loaders[f'{fold_id}_{stage}'] = ItemLoader(meta_data=ds,
                                                            transform=trf, parse_item_cb=parse_item_mnist,
                                                            **kwargs)

        data_provider = DataProvider(item_loaders)

        model = DCGAN(device, nc=1).to(device)

        se = Session(module=model, optimizer=None, loss=None)

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


