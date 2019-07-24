import torch
from collagen.models.segmentation.backbones import ResNetBackbone
import pytest
from .fixtures import tensor_224


@pytest.mark.parametrize('backbone,conf', [('resnet18', 512), ('resnet34', 512), ('resnet50', 2048)])
def test_resnet_backbone_init_forward(backbone, conf, tensor_224):
    model = ResNetBackbone(backbone)
    model.eval()
    with torch.no_grad():
        output = model(tensor_224)

    assert len(output) == 5
    assert output[-1].size(1) == conf
