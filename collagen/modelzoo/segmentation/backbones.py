from torch import nn
import pretrainedmodels
from collagen.core import Module


class ResNetBackbone(Module):
    """
    Extended implementation from https://github.com/qubvel/segmentation_models.pytorch

    """
    def __init__(self, backbone_name='resnet50', dropout=None):
        super(ResNetBackbone, self).__init__()
        self.backbone_name = backbone_name

        backbone = pretrainedmodels.__dict__[backbone_name](num_classes=1000, pretrained='imagenet')
        self.layer0 = nn.Sequential(backbone.conv1,
                                    backbone.bn1,
                                    nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(backbone.maxpool,
                                    backbone.layer1)

        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dropout = dropout

        self.shape_dict = {'resnet18':  (512, 256, 128, 64, 64),
                           'resnet34': (512, 256, 128, 64, 64),
                           'resnet50': (2048, 1024, 512, 256, 64)}

    @property
    def output_shapes(self):
        return self.shape_dict[self.backbone_name]

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4

    def get_features(self):
        pass

    def get_features_by_name(self, name: str):
        pass
