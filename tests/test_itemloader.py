from collagen.data._itemloader import ItemLoader
from collagen.data._dataset import DataFrameDataset

from .fixtures import md_1_dict_6_row_y_class, md_1_df_9_row_y_class, \
    md_1_parse_item, target_parser, md_1_df_6_row_y_class


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

"""
def test_loader_same_shape_sampler_k_1_batch_3(md_1_df_9_row_y_class, md_1_parse_item, img_list):
    root = "./imgs_test_loader_sampler_k_1_batch_3"
    batch_size = 3
    img_list
    itermloader = ItemLoader(root, meta_data=md_1_df_9_row_y_class,
                             batch_size=batch_size, parse_item_cb=md_1_parse_item)
    samples = itermloader.sampler(1)

    batched_labels = np.zeros((3, 6), dtype=np.float)
    batched_labels[0, 0] = 1
    batched_labels[1, 1] = 1
    batched_labels[2, 2] = 1

    # expected_tensor = torch.from_numpy(batched_imgs)
    expected_target = torch.from_numpy(batched_labels)

    expected_output = {'target': expected_target}
    # print("Predicted:\n{}\nGroundtruth:\n{}".format(samples[0], expected_output))
    
    # diff_img = torch.sum(torch.abs(samples[0]['img'] - expected_output['img']))
    diff_target = torch.sum(torch.abs(samples[0]['target'] - expected_output['target']))

    # print("Diff img: {}, diff target: {}".format(diff_img, diff_target))
    assert(diff_target == 0)
"""