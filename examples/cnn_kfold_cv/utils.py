import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import solt
import solt.transforms as slt


def parse_item(root, entry, trf, data_key, target_key):
    img = entry[data_key]

    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    if img.shape[-1] == 1:
        stats = {'mean': (0.1307,), 'std': (0.3081,)}
    elif img.shape[-1] == 3:
        stats = {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.247, 0.243, 0.261)}
    else:
        raise ValueError("Not support channels of {}".format(img.shape[-1]))

    trf_data = trf({'image': img}, normalize=True, **stats)
    return {data_key: trf_data['image'], target_key: entry[target_key]}


def my_transforms():
    train_trf = solt.Stream([
        slt.Scale(range_x=(0.9, 1.1), same=False, p=0.5),
        slt.Shear(range_x=(-0.05, 0.05), p=0.5),
        slt.Rotate((-5, 5), p=0.5),
        slt.Pad(pad_to=(32, 32))
    ])

    test_trf = solt.Stream([slt.Pad(pad_to=(32, 32))])

    return {'train': train_trf, 'eval': test_trf}


class SimpleConvNet(nn.Module):
    def __init__(self, bw, drop_rate=0.2, n_classes=10, n_channels=1):
        super().__init__()
        self.n_filters_last = bw * 2

        self.conv1 = self.make_layer(n_channels, bw)
        self.conv2 = self.make_layer(bw, bw)
        self.conv3 = self.make_layer(bw, bw * 2)
        self.conv4 = self.make_layer(bw * 2, self.n_filters_last)

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(nn.Dropout(drop_rate),
                                        nn.Linear(self.n_filters_last, n_classes))

    @staticmethod
    def make_layer(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 3, 1, 1),
                             nn.BatchNorm2d(out),
                             nn.ReLU(True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # 16x16
        x = self.conv3(x)
        x = self.pool(x)  # 8x8
        x = self.conv4(x)
        x = self.pool(x)  # 4x4

        x = F.adaptive_avg_pool2d(x, 1)

        x = x.view(x.size(0), -1)

        return self.classifier(x)
