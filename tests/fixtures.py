import pytest
import pandas as pd 
import numpy as np
import torch 
from collagen.core import Module
from torch import nn
import torch.nn.functional as F


def gen_5_class_random_data_frame(size):
    np.random.seed(42)
    return pd.DataFrame(data={'fname': [f'img_{i}.png' for i in range(size)],
                              'target': np.random.choice(5, size, replace=True)})


def gen_multi_class_img_minibatch(batch_size, n_channels, height, width, n_classes=2):
    shift = 0.9 / n_channels  # there will be some overlap between the chunks
    batch = []
    targets = []

    for cls in range(n_classes):
        np.random.seed(42 + cls)
        batch_tmp = torch.from_numpy(np.random.rand(batch_size // n_classes,
                                                    n_channels, height, width) / n_classes + shift * n_classes)
        batch.append(batch_tmp)
        targets.append(torch.zeros(batch_size // n_classes).float() + cls)

    if batch_size % 2 != 0:
        np.random.seed(42 + cls)
        batch_tmp = torch.from_numpy(np.random.rand(batch_size // n_classes,
                                                    n_channels, height, width) / n_classes + shift * n_classes)
        batch.append(batch_tmp)
        targets.append(torch.zeros(batch_size // n_classes).float() + cls)

    batch = torch.cat(batch).float()
    if n_classes == 2:
        targets = torch.cat(targets).float()
    elif n_classes > 2:
        targets = torch.cat(targets).long()
    else:
        raise NotImplementedError

    shuffle_ind = torch.randperm(batch_size)
    return batch[shuffle_ind], targets[shuffle_ind].unsqueeze(1)


class DumbNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super(DumbNet, self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(3, n_features, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(n_features),
                                     nn.ReLU(True))

        self.layer_2 = nn.Sequential(nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(n_features),
                                     nn.ReLU(True))

        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        o = self.layer_1(x)
        o = self.layer_2(o)
        features = F.adaptive_avg_pool2d(o, 1).view(x.size(0), -1)

        return self.fc(features)


@pytest.fixture(scope='module', params=[32, 64])
def metadata_fname_target_5_classes(request):
    return gen_5_class_random_data_frame(request.param)


@pytest.fixture
def ones_image_parser():
    def parser(data_dir, entry, transforms, data_key='data', target_key='target'):
        img = np.ones((9, 9))
        target = entry[target_key]

        img, target = transforms(img, target)

        return {'img': img, 'target': target, 'fname': entry.fname}
    return parser


@pytest.fixture
def img_target_transformer():
    def empty_transform(img, target):
        return torch.from_numpy(img.astype(np.float32)).unsqueeze(0), target
    return empty_transform


@pytest.fixture(scope='function', params=((2, 3, 7, 7, 2), (3, 3, 9, 9, 2),
                                          (2, 3, 8, 8, 2), (3, 3, 4, 4, 2)))
def classification_minibatch_two_class(request):
    return gen_multi_class_img_minibatch(request.param[0], request.param[1],
                                         request.param[2], request.param[3],
                                         request.param[4])

@pytest.fixture(scope='function', params=((2, 3, 7, 7, 3), (3, 3, 9, 9, 3),
                                          (2, 3, 8, 8, 5), (3, 3, 4, 4, 5)))
def classification_minibatch_multi_class(request):
    return gen_multi_class_img_minibatch(request.param[0], request.param[1],
                                         request.param[2], request.param[3],
                                         request.param[4])

@pytest.fixture(scope='function', params=((2, 3, 7, 7, 3, 2), (3, 3, 9, 9, 3, 3),
                                          (2, 3, 8, 8, 5, 2), (3, 3, 4, 4, 5, 3)))
def classification_minibatches_seq_multiclass(request):
    s = []
    for _ in range(request.param[5]):
        s.append(gen_multi_class_img_minibatch(request.param[0], request.param[1],
                                               request.param[2], request.param[3],
                                               request.param[4]))
    return s


@pytest.fixture(scope='function', params=((2, 3, 7, 7, 2, 2), (3, 3, 9, 9, 2, 3),
                                          (2, 3, 8, 8, 2, 2), (3, 3, 4, 4, 2, 3)))
def classification_minibatches_seq_two_class(request):
    s = []
    for _ in range(request.param[5]):
        s.append(gen_multi_class_img_minibatch(request.param[0], request.param[1],
                                               request.param[2], request.param[3],
                                               request.param[4]))
    return s


@pytest.fixture(scope='module')
def dumb_net():
    return DumbNet


@pytest.fixture(scope='function')
def tensor_224():
    return torch.FloatTensor(1, 3, 224, 224)

