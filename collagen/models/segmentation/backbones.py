from torch import nn
import pretrainedmodels


class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(ResNetBackbone, self).__init__()
        backbone = pretrainedmodels.__dict__[backbone_name](num_classes=1000, pretrained='imagenet')
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4
