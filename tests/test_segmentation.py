from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.modelzoo.segmentation.backbones import ResNetBackbone
from collagen.modelzoo.segmentation.decoders import FPNDecoder

from .fixtures import *


@pytest.mark.parametrize('backbone,conf', [('resnet18', 512), ('resnet34', 512), ('resnet50', 2048)])
def test_resnet_backbone_init_forward(backbone, conf, tensor_224):
    torch.manual_seed(42)
    model = ResNetBackbone(backbone)
    model.eval()
    with torch.no_grad():
        output = model(tensor_224)

    assert len(output) == 5
    assert output[-1].size(1) == conf


@pytest.mark.parametrize('normalization', ['BN', 'IN'])
@pytest.mark.parametrize('bayesian_drop', [None, 0.2])
@pytest.mark.parametrize('spatial_drop', [None, 0.2])
@pytest.mark.parametrize('backbone', ['resnet18', ])
def test_resnet_fpn_returns_the_same_size(backbone, spatial_drop, bayesian_drop, normalization, tensor_224):
    torch.manual_seed(42)
    model = EncoderDecoder(1, backbone=backbone, decoder='FPN',
                           bayesian_dropout=bayesian_drop,
                           decoder_normalization=normalization, spatial_dropout=spatial_drop)
    model.eval()
    with torch.no_grad():
        output = model(tensor_224)

    assert output.size()[-2:] == tensor_224.size()[-2:]


@pytest.mark.parametrize('dropout,represenetation_differ', [(None, False), (0.9, True)])
@pytest.mark.parametrize('backbone', ['resnet18', ])
def test_backbone_dropout_on_off(backbone, dropout, tensor_224, represenetation_differ):
    torch.manual_seed(42)
    model = ResNetBackbone(backbone, dropout=dropout)
    model.eval()
    model.switch_dropout()
    with torch.no_grad():
        output1 = model(tensor_224)
        output2 = model(tensor_224)

    err = 0

    for l1, l2 in zip(output1, output2):
        err += torch.pow(l1 - l2, 2).sum().item()

    difference_test = err > 1e-6
    assert difference_test == represenetation_differ


@pytest.mark.parametrize('dropout,represenetation_differ', [(None, False), (0.2, True)])
def test_decoder_dropout_on_off(dropout, tensor_224, represenetation_differ):
    torch.manual_seed(42)
    backbone = 'resnet18'
    enc_model = ResNetBackbone(backbone)
    dec_model = FPNDecoder(encoder_channels=enc_model.output_shapes, bayesian_dropout=dropout)

    enc_model.eval()
    dec_model.eval()
    dec_model.switch_dropout()
    with torch.no_grad():
        enc_out = enc_model(tensor_224)
        output1 = dec_model(enc_out)
        output2 = dec_model(enc_out)

    difference_test = torch.pow(output2 - output1, 2).sum().item() > 1e-6
    assert difference_test == represenetation_differ
