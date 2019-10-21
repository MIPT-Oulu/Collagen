import numpy as np
import pandas as pd
import torch
import torch.utils.data.distributed
try:  # Handling API difference between pytorch 1.1 and 1.2
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate


from ._dataset import DataFrameDataset


class DistributedItemLoader(object):
    """Combines DataFrameDataset and DataLoader, and provides single- or multi-process iterators over the dataset.

    """

    def __init__(self, meta_data: pd.DataFrame or None = None, parse_item_cb: callable or None = None, args=None,
                 root: str or None = None,
                 collate_fn: callable = default_collate, transform: callable or None = None,
                 drop_last: bool = False, timeout: int = 0, name: str = "loader"):
        """Parameters
        ----------
        meta_data : pandas.DataFrame or None
            Meta data of data and labels.
        parse_item_cb : callable or None
            Parses each row of :attr:meta_data`.
        args: Namespace
            Program arguments passed or parsed from the main file or function
        root : str
            Path to root directory of input data. Default is None (empty)
        collate_fn : callable, optional
            Merges a list of samples to form a mini-batch. (the default is None)
        transform : callable, optional
            Transforms row of :attr:`meta_data`. (the default is None)
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (the default is False)
        timeout : int, optional
            If positive, the timeout value for collecting a batch from workers.
            If ``0``, ignores ``timeout`` notion. Must be non-negative. (the default is 0)
        """
        batch_size = args.batch_size
        num_workers = args.workers

        if root is None:
            root = ''

        self.__name = name

        if meta_data is None:
            self.__dataset = None
        else:
            self.__dataset = DataFrameDataset(root, meta_data=meta_data,
                                              parse_item_cb=parse_item_cb, transform=transform)
        if self.__dataset is None:
            self.__data_loader = None
        else:
            self.__sampler = torch.utils.data.distributed.DistributedSampler(self.__dataset, rank=args.gpu,
                                                                             num_replicas=args.world_size)
            self.__data_loader = torch.utils.data.DataLoader(dataset=self.__dataset,
                                                             batch_size=batch_size,
                                                             shuffle=(self.__sampler is None),
                                                             sampler=self.__sampler,
                                                             num_workers=num_workers,
                                                             collate_fn=collate_fn,
                                                             pin_memory=True, # pin memory needs to be true for DDP
                                                             drop_last=drop_last,
                                                             timeout=timeout)

        self.__transform = transform
        self.drop_last: bool = drop_last
        self.batch_size: int = batch_size
        self.__iter_loader = None
        self.parse_item = parse_item_cb

    @property
    def transform(self):
        return self.__transform

    def __len__(self):
        """ Get length of the dataloader.
        """
        return len(self.__data_loader)

    def sample(self, k=1):
        """Sample one or more mini-batches.

        Parameters
        ----------
        k : int, optional
            The number of batches to sample. (the default is 1)
        
        Returns
        -------
        samples : list
            List of sampled batches.
        """
        samples = []
        for i in range(k):
            try:
                if self.__iter_loader is None:
                    self.__iter_loader = iter(self.__data_loader)
                batch = next(self.__iter_loader)
            except StopIteration:
                del self.__iter_loader
                self.__iter_loader = iter(self.__data_loader)
                batch = next(self.__iter_loader)

            batch['name'] = self.__name
            samples.append(batch)

        return samples

