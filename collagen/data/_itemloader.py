import sys
import os
import numpy as numpy
import pandas as pd
import torchvision
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
from ._dataset import DataFrameDataset

class ItemLoader(object):
    """ItemLoader. Combines DataFrameDataset and DataLoader, and provides single- or multi-process iterators over the dataset.
        
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
        Tranforms row of :attr:`meta_data` (the default is None)
    sampler : Sampler, optional
        Defines the strategy to draw samples from
        the dataset. If specified, ``shuffle`` must be False. (the default is None)
    batch_sampler : [type], optional
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
    worker_init_fn : [type], optional
        If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (the default is None)
    
    Note
    ----
    By default, each worker will have its PyTorch seed set to
    ``base_seed + worker_id``, where ``base_seed`` is a long generated
    by main process using its RNG. However, seeds for other libraies
    may be duplicated upon initializing workers (w.g., NumPy), causing
    each worker to return identical random numbers. (See
    :ref:`dataloader-workers-random-seed` section in FAQ.) You may
    use :func:`torch.initial_seed()` to access the PyTorch seed for
    each worker in :attr:`worker_init_fn`, and use it to set other
    seeds before data loading.
    
    """

    def __init__(self, root, meta_data, parse_item_cb, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, collate_fn=None, transform=None,
                sampler=None, batch_sampler=None, drop_last=True, timeout=0, worker_init_fn=None):

        self.dataset = DataFrameDataset(root, meta_data=meta_data, parse_item_cb=parse_item_cb)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, 
                                                        batch_size=batch_size, 
                                                        shuffle=shuffle,
                                                        sampler=sampler,
                                                        batch_sampler=batch_sampler,
                                                        num_workers=num_workers,
                                                        # collate_fn=collate_fn,
                                                        pin_memory=pin_memory,
                                                        drop_last=drop_last,
                                                        timeout=timeout,
                                                        worker_init_fn=worker_init_fn)
        self.iter_batch_sampler = iter(self.data_loader)
        self.drop_last = drop_last
        self.batch_size = batch_size

        if collate_fn is not None:
            self.data_loader.collate_fn=collate_fn
        
    def __len__(self):
        """ Get length of dataset
        """
        return len(self.dataset)

    def sampler(self, k=1):
        """Sample a number of batches
        
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
                batch = next(self.iter_batch_sampler)
            except StopIteration:
                self.iter_batch_sampler = iter(self.data_loader)
                continue
            
            if not self.drop_last or batch["img"].shape[0] == self.batch_size:
                samples.append(batch)
            
        return samples
