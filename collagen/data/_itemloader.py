import pandas as pd
import torch
from torch.utils.data.sampler import SequentialSampler

from torch.utils.data.dataloader import default_collate

from ._dataset import DataFrameDataset
import numpy as np


class ItemLoader(object):
    """Combines DataFrameDataset and DataLoader, and provides single- or multi-process iterators over the dataset.
        
    Parameters
    ----------
    root : str
        Path to root directory of input data
    meta_data : pandas.DataFrame
        Meta data of data and labels
    parse_item_cb : callable
        Parses each row of :attr:meta_data`
    batch_size : int, optional
        How many data per batch to load (the default is 1)
    num_workers : int, optional
        How many subprocesses to use for data
        loading. 0 means that the data will be loaded in the main process. (the default is 0)
    shuffle : bool, optional
        Set to ``True`` to have the data reshuffled
        at every epoch (the default is False)
    pin_memory : bool, optional
        If ``True``, the data loader will copy tensors
        into CUDA pinned memory before returning them. (the default is False)
    collate_fn : callable, optional
        Merges a list of samples to form a mini-batch. (the default is None)
    transform : callable, optional
        Transforms row of :attr:`meta_data` (the default is None)
    sampler : Sampler, optional
        Defines the strategy to draw samples from
        the dataset. If specified, ``shuffle`` must be False. (the default is None)
    batch_sampler : callable, optional
        like sampler, but returns a batch of
        indices at a time. Mutually exclusive with :attr:`batch_size`,
        :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`. (the default is None)
    drop_last : bool, optional
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (the default is True)
    timeout : int, optional
        if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (the default is 0)
    """

    def __init__(self, root: str, meta_data: pd.DataFrame, parse_item_cb: callable, batch_size: int = 1,
                 num_workers: int = 0, shuffle: bool = False, pin_memory: bool = False,
                 collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None,
                 batch_sampler=None, drop_last: bool = True, timeout: int = 0):

        self.dataset = DataFrameDataset(root, meta_data=meta_data, parse_item_cb=parse_item_cb, transform=transform)
        self.data_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       sampler=sampler,
                                                       batch_sampler=batch_sampler,
                                                       num_workers=num_workers,
                                                       collate_fn=collate_fn,
                                                       pin_memory=pin_memory,
                                                       drop_last=drop_last,
                                                       timeout=timeout,
                                                       worker_init_fn=lambda wid: np.random.seed(np.uint32(
                                                           torch.initial_seed() + wid)))

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.__iter_loader = iter(self.data_loader)
        
    def __len__(self):
        """ Get length of teh dataloader

        """
        return len(self.data_loader)

    def sample(self, k=1):
        """Sample one or more mini-batches.

        if not
        
        Parameters
        ----------
        k : int, optional
            The number of batches to sample (the default is 1)
        
        Returns
        -------
        list
            List of sampled batches
        """
        samples = []
        for i in range(k):
            try:
                batch = next(self.__iter_loader)
            except StopIteration:
                del self.__iter_loader
                self.__iter_loader = iter(self.data_loader)
                batch = next(self.__iter_loader)

            samples.append(batch)
            
        return samples
