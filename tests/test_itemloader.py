from collagen.data._itemloader import ItemLoader
import itertools
from collagen.data._dataset import DataFrameDataset

from .fixtures import img_target_transformer, ones_image_parser, img_target_transformer, \
    metadata_fname_target_5_classes


import torch
import pytest
import numpy as np


@pytest.mark.parametrize('batch_size, n_samples', itertools.product([32, 11, 3], [1, 3, 6]))
def test_loader_samples_batches(batch_size, n_samples, metadata_fname_target_5_classes,
                                ones_image_parser, img_target_transformer):

    itermloader = ItemLoader('/tmp/', meta_data=metadata_fname_target_5_classes,
                             batch_size=batch_size, parse_item_cb=ones_image_parser,
                             transform=img_target_transformer, shuffle=True)

    samples = itermloader.sample(n_samples)

    assert len(samples) == n_samples
    assert samples[0]['img'].size(0) == batch_size
    assert samples[0]['target'].size(0) == batch_size


@pytest.mark.parametrize('batch_size, n_samples', itertools.product([32], [20, 100]))
def test_loader_endless_sampling_works(batch_size, n_samples, metadata_fname_target_5_classes,
                                ones_image_parser, img_target_transformer):

    itermloader = ItemLoader('/tmp/', meta_data=metadata_fname_target_5_classes,
                             batch_size=batch_size, parse_item_cb=ones_image_parser,
                             transform=img_target_transformer, shuffle=True)

    for i in range(2*len(itermloader)):
        samples = itermloader.sample(n_samples)

        assert len(samples) == n_samples
        assert samples[0]['img'].size(0) == batch_size
        assert samples[0]['target'].size(0) == batch_size


@pytest.mark.parametrize('batch_size, n_samples, drop_last', itertools.product([3, 32], [1, 2], [True, False]))
def test_loader_drop_last(batch_size, n_samples, metadata_fname_target_5_classes,
                          ones_image_parser, img_target_transformer, drop_last):

    itermloader = ItemLoader('/tmp/', meta_data=metadata_fname_target_5_classes,
                             batch_size=batch_size, parse_item_cb=ones_image_parser,
                             transform=img_target_transformer, shuffle=True, drop_last=drop_last)

    if drop_last:
        assert len(itermloader) == metadata_fname_target_5_classes.shape[0] // batch_size
    else:
        if metadata_fname_target_5_classes.shape[0] % batch_size != 0:
            assert len(itermloader) == metadata_fname_target_5_classes.shape[0] // batch_size + 1
        else:
            assert len(itermloader) == metadata_fname_target_5_classes.shape[0] // batch_size

