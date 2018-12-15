from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np 
import torch
import os

DEFAULT_FILE_NAME_KEY = "file_name"

class DataFrameDataset(Dataset):
    """A class representing a DataFrameDataset.
    """    

    def __init__(self, root, meta_data, parse_item_cb, transform=None, rgb=True):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(meta_data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))
        self.root = root
        self.meta_data = meta_data
        self.rgb = rgb
        # print("meta_data:\n{}\nlen:{}".format(self.meta_data, self.__len__()))
        self.parse_item_cb = parse_item_cb

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: dictionary of `index`-th row
        """
        entry = self.meta_data.iloc[index]
        entry = self.parse_item_cb(entry.to_dict())
        print("entry: {}".format(entry))
        assert(isinstance(entry, dict))
        file_fullname = os.path.join(self.root, entry[DEFAULT_FILE_NAME_KEY])
        pil_img = Image.open(file_fullname)
        if not self.rgb:
            pil_img = pil_img.convert("L")
        
        pt_tensor_img = torch.from_numpy(np.array(pil_img))

        # print("entry: shape: {}, \n{}".format(pt_tensor_img.shape, entry))
        processed_entry = {"img": pt_tensor_img}
        for key in entry:
            if key != DEFAULT_FILE_NAME_KEY:
                processed_entry[key] = entry[key]

        if not isinstance(processed_entry, dict):
            raise TypeError("`entry` must be `dict`")
        return processed_entry

    def _default_parse_item(self, entry, rgb=True):
        pil_img = Image.open(entry["file_name"])
        if not rgb:
            pil_img = pil_img.convert('L')

        return 1
        

    def __len__(self):
        return len(self.meta_data.index)
