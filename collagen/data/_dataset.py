from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
import torch
import os

class DataFrameDataset(Dataset):
    """Dataset based on ``pandas.DataFrame``
        
    Parameters
    ----------
    root : str
        Path to root directory of input data
    meta_data : pandas.DataFrame
        Meta data of data and labels
    parse_item_cb : callable
        Callback function to parse each row of :attr:`meta_data`
    transform : callable, optional
        Transformation applied to row of :attr:`meta_data` (the default is None, which means to do nothing)        
    
    Raises
    ------
    TypeError
        `root` must be `str`
    TypeError
        `meta_data` must be `pandas.DataFrame`
    
    """
    def __init__(self, root, meta_data, parse_item_cb, transform=None):
        if not isinstance(root, str):
            raise TypeError("`root` must be `str`")
        if not isinstance(meta_data, pd.DataFrame):
            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))
        self.root = root
        self.meta_data = meta_data
        self.parse_item_cb = parse_item_cb

    def __getitem__(self, index):
        """Get ``index``-th parsed item of :attr:`meta_data`
        
        Parameters
        ----------
        index : int
            Index of row
        
        Returns
        -------
        dict
            dictionary of `index`-th parsed item
        """
        entry = self.meta_data.iloc[index]
        entry = self.parse_item_cb(self.root, entry.to_dict())
        assert(isinstance(entry, dict))
        return entry

    def __len__(self):
        """Get length of `meta_data`
        """
        return len(self.meta_data.index)
