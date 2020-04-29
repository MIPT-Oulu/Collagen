from torch import nn
import torch
import solt
import solt.transforms as slt

from collagen.core.utils import to_cpu


def parse_item(root, entry, trf, data_key, target_key):
    img = entry[data_key]

    stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}

    trf_data = trf({'image': img}, normalize=True, **stats)
    return {data_key: trf_data['image'], target_key: entry[target_key]}


def my_transforms():
    train_trf = solt.Stream([
        slt.Scale(range_x=(0.9, 1.1), range_y=(0.9, 1.1), same=True, p=0.5),
        slt.Translate(range_x=(-0.05, 0.05), range_y=(-0.05, 0.05), p=0.5),
        slt.Rotate((-5, 5), p=0.5),
        slt.GammaCorrection(gamma_range=0.1, p=0.5),
        slt.Noise(gain_range=0.1, p=0.8)
    ])

    test_trf = solt.Stream([])

    custom_trf = solt.Stream([
        slt.Scale(range_x=(0.9, 1.1), range_y=(0.9, 1.1), same=True, p=0.5),
        slt.Translate(range_x=(-0.05, 0.05), range_y=(-0.05, 0.05), p=0.5),
        slt.Rotate((-5, 5), p=0.5),
        slt.GammaCorrection(gamma_range=0.1, p=0.5),
        slt.Noise(gain_range=0.1, p=0.8)
    ])

    return {'train': train_trf, 'eval': test_trf, 'transforms': custom_trf}


def data_rearrange(x, y):
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)
    return x[index, :, :, :], y[index]


def cond_accuracy_meter(target, output):
    return target['name'].startswith('l')


def parse_target_accuracy_meter(target):
    if target['name'].startswith('l'):
        return target['target']
    else:
        return None


def parse_class(x):
    if isinstance(x, dict) and x['name'].startswith('l'):
        x = x['target']
    elif isinstance(x, torch.Tensor):
        pass
    else:
        return None

    if x is None:
        return None
    elif len(x.shape) == 2:
        output_cpu = to_cpu(x.argmax(dim=1), use_numpy=True)
    elif len(x.shape) == 1:
        output_cpu = to_cpu(x, use_numpy=True)
    else:
        raise ValueError("Only support dims 1 or 2, but got {}".format(len(x.shape)))
    output_cpu = output_cpu.astype(int)
    return output_cpu
