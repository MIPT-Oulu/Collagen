# Some functions are copied and/or modified from cnn examples
import argparse
import torch
import solt.data as sld
import solt.transforms as slt

from collagen.data.utils import ApplyTransform, Normalize, Compose
from collagen.data import ItemLoader, DataProvider
from collagen.callbacks import RunningAverageMeter
from collagen.callbacks import ScalarMeterLogger
from collagen.callbacks import ModelSaver


def parse_item_ae(root, entry, trf, data_key, target_key):
    img, target = trf(entry[data_key])
    return {data_key: img, target_key: target}


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bw', type=int, default=64, help='Bandwidth of model')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for data loader')
    parser.add_argument('--save_data', default='data', help='Where to save downloaded dataset')
    parser.add_argument('--snapshots', default='snapshots', help='Where to save the snapshots')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset name')
    parser.add_argument('--device', type=str, default="cuda:0", help='Use `cuda` or `cpu`')
    parser.add_argument('--data_dir', type=str, default="data", help='Data directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--comment', type=str, default="cnn", help='Comment of log')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    args = parser.parse_args()

    return args


def wrap2solt(inp):
    return sld.DataContainer(inp, 'I')


def unpack_solt(dc: sld.DataContainer):
    img = dc.data[0].squeeze()
    img = torch.from_numpy(img).unsqueeze(0).float().div(255.)
    img, target = torch.stack((img, img, img), 0), torch.stack((img, img, img), 0)
    return img.squeeze(), target.squeeze()


def init_mnist_transforms():
    norm_mean_std = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(norm_mean_std, [0, 1])
    ])

    test_trf = Compose([
        wrap2solt,
        slt.PadTransform(pad_to=32),
        unpack_solt,
        ApplyTransform(norm_mean_std, [0, 1])
    ])

    return train_trf, test_trf


def cast2int(data):
    """
    cast2int type casts data to integer
    :param data: any numeric value
    :return: integer of tensor or other type
    """
    if isinstance(data, int):
        return data
    else:
        return data.item()


def init_data_provider(args, df_train, df_val):
    item_loaders = dict()
    for stage, df in zip(['train', 'eval'], [df_train, df_val]):
        item_loaders[f'mnist_{stage}'] = ItemLoader(meta_data=df,
                                                    transform=init_mnist_transforms()[0],
                                                    parse_item_cb=parse_item_ae,
                                                    batch_size=args.bs,
                                                    num_workers=args.num_threads,
                                                    shuffle=True if stage == 'train' else False)
    return DataProvider(item_loaders)


def init_callbacks(args, summary_writer, model):
    train_cbs = (RunningAverageMeter(prefix='train', name='loss'))
    val_cbs = (RunningAverageMeter(prefix='eval', name='loss'),
               ScalarMeterLogger(writer=summary_writer),
               ModelSaver(metric_names='eval/loss', save_dir=args.snapshots, conditions='min', model=model))
    return {'train': train_cbs, 'eval': val_cbs}
