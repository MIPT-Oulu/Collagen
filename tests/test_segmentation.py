import torch
from collagen.modelzoo.segmentation.backbones import ResNetBackbone
from collagen.modelzoo.segmentation import EncoderDecoder
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


@pytest.mark.parametrize('normalization', ['BN', 'IN'])
@pytest.mark.parametrize('bayesian_drop', [None, 0.2])
@pytest.mark.parametrize('spatial_drop', [None, 0.2])
@pytest.mark.parametrize('backbone', ['resnet18', 'resnet34', 'resnet50'])
def test_resnet_fpn_returns_the_same_size(backbone, spatial_drop, bayesian_drop, normalization, tensor_224):
    model = EncoderDecoder(1, backbone=backbone, decoder='FPN',
                           bayesian_dropout=bayesian_drop,
                           decoder_normalization=normalization, spatial_dropout=spatial_drop)
    model.eval()
    with torch.no_grad():
        output = model(tensor_224)

    assert output.size()[-2:] == tensor_224.size()[-2:]
