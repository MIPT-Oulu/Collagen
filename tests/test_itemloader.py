from collagen.data._itemloader import ItemLoader
import itertools
from collagen.data._dataset import DataFrameDataset

from .fixtures import md_1_dict_6_row_y_class, md_1_df_9_row_y_class, md_1_parse_item, \
    ones_image_parser, md_1_df_6_row_y_class, metadata_fname_target_5_classes, img_target_transformer


import torch
import pytest
import numpy as np


def test_valid_type_root_int(md_1_df_6_row_y_class):
    with pytest.raises(TypeError):
        ItemLoader(1, md_1_df_6_row_y_class)


def test_valid_type_root_list(md_1_df_6_row_y_class):
    with pytest.raises(TypeError):
        ItemLoader([], md_1_df_6_row_y_class)


def test_valid_type_metadata_1():
    with pytest.raises(TypeError):
        ItemLoader(".", 1)


def test_not_enough_arg_input_0_arg():
    with pytest.raises(TypeError):
        ItemLoader()


def test_not_enough_arg_input_1_arg():
    with pytest.raises(TypeError):
        ItemLoader(".")


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
