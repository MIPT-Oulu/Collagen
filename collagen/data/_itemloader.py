import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import default_collate

from ._dataset import DataFrameDataset


class ItemLoader(object):
    """Combines DataFrameDataset and DataLoader, and provides single- or multi-process iterators over the dataset.
        
    Parameters
    ----------
    meta_data : pandas.DataFrame or None
        Meta data of data and labels.
    parse_item_cb : callable or None
        Parses each row of :attr:meta_data`.
    batch_size : int, optional
        How many data per batch to load. (the default is 1)
    root : str
        Path to root directory of input data. Default is None (empty)
    num_workers : int, optional
        How many subprocesses to use for data loading. If equals to 0, 
        the data will be loaded in the main process. (the default is 0)
    shuffle : bool, optional
        Set to ``True`` to have the data reshuffled at every epoch. (the default is False)
    pin_memory : bool, optional
        If ``True``, the data loader will copy tensors into CUDA pinned memory
        before returning them. (the default is False)
    collate_fn : callable, optional
        Merges a list of samples to form a mini-batch. (the default is None)
    transform : callable, optional
        Transforms row of :attr:`meta_data`. (the default is None)
    sampler : Sampler, optional
        Defines the strategy to draw samples from the dataset. If specified,
        ``shuffle`` must be False. (the default is None)
    batch_sampler : callable, optional
        Like sampler, but returns a batch of indices at a time.
        Mutually exclusive with :attr:`batch_size`, :attr:`shuffle`,
        :attr:`sampler`, and :attr:`drop_last`. (the default is None)
    drop_last : bool, optional
        Set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (the default is True)
    timeout : int, optional
        If positive, the timeout value for collecting a batch from workers.
        If ``0``, ignores ``timeout`` notion. Must be non-negative. (the default is 0)
    """

    def __init__(self, meta_data: pd.DataFrame or None = None, parse_item_cb: callable or None = None,
                 root: str or None = None, batch_size: int = 1,
                 num_workers: int = 0, shuffle: bool = False, pin_memory: bool = False,
                 collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None,
                 batch_sampler=None, drop_last: bool = True, timeout: int = 0):
        if root is None:
            root = ''

        if meta_data is None:
            self.__dataset = None
        else:
            self.__dataset = DataFrameDataset(root, meta_data=meta_data,
                                              parse_item_cb=parse_item_cb, transform=transform)
        if self.__dataset is None:
            self.__data_loader = None
        else:
            self.__data_loader = torch.utils.data.DataLoader(self.__dataset,
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

        self.drop_last: bool = drop_last
        self.batch_size: int = batch_size
        self.__iter_loader = None
        self.parse_item = parse_item_cb
        
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

            samples.append(batch)

        return samples


class GANFakeSampler(ItemLoader):
    def __init__(self, g_network, batch_size, latent_size):
        super().__init__(meta_data=None, parse_item_cb=None)
        self.__latent_size = latent_size
        self.batch_size = batch_size
        self.__g_network = g_network

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__g_network.parameters()).device)
            fake: torch.Tensor = self.__g_network(noise_on_device)
            samples.append({'data': fake.detach(), 'target': torch.zeros(self.batch_size).to(fake.device), 'latent': noise})

        return samples

    def __len__(self):
        return 1


class SSGANFakeSampler(ItemLoader):
    def __init__(self, g_network, batch_size, latent_size, n_classes):
        super().__init__(meta_data=None, parse_item_cb=None)
        self.__latent_size = latent_size
        self.batch_size = batch_size
        self.__g_network = g_network
        self.__n_classes = n_classes

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__g_network.parameters()).device)
            # freeze_modules(modules=self.__g_network)
            fake: torch.Tensor = self.__g_network(noise_on_device)
            # freeze_modules(modules=self.__g_network, invert=True)
            target = torch.zeros([self.batch_size, self.__n_classes + 1]).to(fake.device)
            target[:,-2] = 1.0
            samples.append({'data': fake.detach(), 'target': target, 'latent': noise, 'valid': target[:,-1]})

        return samples

    def __len__(self):
        return 1


class GaussianNoiseSampler(ItemLoader):
    def __init__(self, batch_size, latent_size, device, n_classes):
        super().__init__(meta_data=None, parse_item_cb=None)
        self.latent_size = latent_size
        self.device = device
        self.batch_size = batch_size
        self.__n_classes = n_classes

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
            target = torch.zeros([self.batch_size, self.__n_classes + 1]).to(self.device)
            samples.append({'latent': noise, 'target': target})

        return samples

    def __len__(self):
        return 1