from ..collagen.data._itemloader import ItemLoader
from ..collagen.data._dataset import DataFrameDataset
from .fixtures import *
from PIL import Image
import os
from shutil import rmtree
import torch
import pytest

def test_valid_type_root_int():
    with pytest.raises(TypeError):
        ItemLoader(1, md_1_df_6_row_y_class())

def test_valid_type_root_list():
    with pytest.raises(TypeError):
        ItemLoader([], md_1_df_6_row_y_class())

def test_valid_type_metadata_1():
    df = md_1_df_6_row_y_class()
    root = "."
    with pytest.raises(TypeError):
        ItemLoader(root, 1)    

def test_not_enough_arg_input_0_arg():
    df = md_1_df_6_row_y_class()
    root = "."
    with pytest.raises(TypeError):
        ItemLoader()

def test_not_enough_arg_input_1_arg():
    df = md_1_df_6_row_y_class()
    root = "."
    with pytest.raises(TypeError):
        ItemLoader(root)

def test_loader_same_shape_sampler_k_1_batch_3():
    root = "./imgs_test_loader_sampler_k_1_batch_3"
    batch_size = 3
    if os.path.exists(root):
        rmtree(root)
    os.makedirs(root)
    list_imgs = []
    for i in range(9):
        file_name = "img" + str(i+1) + ".png"
        file_fullname = os.path.join(root, file_name)
        img = np.zeros((2, 9), dtype=int)
        img[:,i] = 1
        list_imgs.append(np.copy(img))
        pil_img = Image.fromarray(img)
        pil_img.save(file_fullname)
    
    itermloader = ItemLoader(root, meta_data=md_1_df_9_row_y_class(), batch_size=batch_size, parse_item_cb=md_1_parse_item)
    samples = itermloader.sampler(1)

    # Prepare result
    # X
    batched_imgs = np.stack(list_imgs[:batch_size], axis=0).astype(np.int32)
    # y
    batched_labels = np.zeros((3, 6), dtype=np.float)
    batched_labels[0,0] = 1
    batched_labels[1,1] = 1
    batched_labels[2,2] = 1

    expected_tensor = torch.from_numpy(batched_imgs)
    expected_target = torch.from_numpy(batched_labels)

    expected_output = {'img': expected_tensor, 'target': expected_target}
    # print("Predicted:\n{}\nGroundtruth:\n{}".format(samples[0], expected_output))
    
    diff_img = torch.sum(torch.abs(samples[0]['img'] - expected_output['img']))
    diff_target = torch.sum(torch.abs(samples[0]['target'] - expected_output['target']))

    # print("Diff img: {}, diff target: {}".format(diff_img, diff_target))
    assert(diff_img == 0 and diff_target == 0)