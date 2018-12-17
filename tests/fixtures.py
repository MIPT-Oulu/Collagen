import pytest
import pandas as pd 
import numpy as np 
import os
import cv2
import torch 


@pytest.fixture
def md_1_dict_6_row_y_class():
    """
    Generates a 6-row dict whose y includes only `class`
    Returns
    -------
    out : dict
    """
    return {'file_name': {0: 'img1.png', 1: 'img2.png', 2: 'img3.png', 3: 'img4.png', 4: 'img5.png', 5: 'img6.png', 6: 'img5.png'}, 'class': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}}


@pytest.fixture
def md_1_dict_9_row_y_class():
    """
    Generates a 6-row dict whose y includes only `class`
    Returns
    -------
    out : dict
    """
    return {'file_name': {0: 'img1.png', 1: 'img2.png', 2: 'img3.png', 3: 'img4.png', 4: 'img5.png', 5: 'img6.png', 6: 'img5.png', 7: 'img4.png' , 8: 'img3.png'}, 'class': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'E', 7: 'D', 8: 'C'}}


@pytest.fixture
def md_1_df_6_row_y_class():
    """
    Generates a 6-row dataframe whose y includes only `class`
    Returns
    -------
    out : pandas.DataFrame
    """
    return pd.DataFrame.from_dict(md_1_dict_6_row_y_class())


@pytest.fixture
def md_1_df_9_row_y_class():
    """
    Generates a 9-row dataframe whose y includes only `class`
    Returns
    -------
    out : pandas.DataFrame
    """
    return pd.DataFrame.from_dict(md_1_dict_9_row_y_class())


@pytest.fixture
def md_1_dict_retrieved_batch_1():
    """
    Generates 1st-row dict of md_1_dict_6_row_y_class
    Returns
    -------
    out : dict
    """
    return {'file_name': ['img1.png'], 'class': ['A']}


@pytest.fixture
def md_1_dict_retrieved_batch_2():
    """
    Generates dict of 1st 2 rows of md_1_dict_6_row_y_class
    Returns
    -------
    out : dict
    """
    return {'file_name': ['img1.png', 'img2.png'], 'class': ['A', 'B']}


@pytest.fixture
def md_1_dict_retrieved_batch_3_iter_1():
    """
    Generates dict of 2st 3 rows of md_1_dict_6_row_y_class
    Returns
    -------
    out : dict
    """
    return [{'file_name': ['img1.png', 'img2.png', 'img3.png'], 'class': ['A', 'B', 'C']}, {'file_name': ['img4.png', 'img5.png', 'img6.png'], 'class': ['D', 'E', 'F']}]


@pytest.fixture
def md_1_parse_item():
    def parser(root, entry):
        classes = ["A", "B", "C", "D", "E", "F"]
        num_cls = len(classes)
        one_hot = np.array([0]*num_cls).astype(np.float)
        clas_to_ind = {}
        for i, l in enumerate(classes):
            clas_to_ind[l] = i

        file_fullname = os.path.join(root, entry["file_name"])
        assert os.path.exists(file_fullname)

        img = cv2.imread(file_fullname, cv2.IMREAD_GRAYSCALE)

        pt_tensor_img = torch.from_numpy(img.astype(np.int32))

        one_hot[clas_to_ind[entry["class"]]] = 1.0

        processed_entry = {"target": one_hot, "img": pt_tensor_img}
        for key in entry:
            if key not in ["class", "img"]:
                processed_entry[key] = entry[key]
        return processed_entry
    return parser


@pytest.fixture
def target_parser():
    def parser(data_dir, entry):
        return {'img': np.ones((9, 9), dtype=np.uint8), 'target': entry.target, 'fname': entry.fname}
    return parser



