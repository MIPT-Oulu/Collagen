import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import default_collate

from ._dataset import DataFrameDataset
from collagen.core.utils import to_cpu


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

            samples.append(batch)

        return samples


class AugmentedGroupSampler(ItemLoader):
    def __init__(self, model: nn.Module, name: str, augmentation, n_augmentations=1, data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = True, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__name = name
        self.__model: nn.Module = model
        self.__n_augmentations = n_augmentations
        self.__augmentation = augmentation
        self.__data_key = data_key
        self.__target_key = target_key

    def sample(self, k=1):
        samples = []
        sampled_rows = super().sample(k)
        for i in range(k):
            imgs = sampled_rows[i][self.__data_key]
            target = sampled_rows[i][self.__target_key]
            list_features = []

            for b in range(imgs.shape[0]):
                list_imgs = [imgs[b, :, :, :]]
                for j in range(self.__n_augmentations):
                    img = imgs[b, :, :, :]
                    if img.shape[0] == 1:
                        img = img[0, :, :]
                    else:
                        img = img.permute(1, 2, 0)

                    img_cpu = to_cpu(img, use_numpy=True)
                    aug_img = self.__augmentation(img_cpu)
                    list_imgs.append(aug_img)
                batch_imgs = torch.stack(list_imgs, dim=0).to(next(self.__model.parameters()).device)
                features = self.__model.get_features(batch_imgs)
                list_features.append(features)
            samples.append({'name': self.__name, 'features': torch.stack(list_features, dim=1), 'data': imgs, 'target': target})
        return samples


class AugmentedGroupStudentTeacherSampler(ItemLoader):
    def __init__(self, name:str, teacher_model: nn.Module, student_model: nn.Module, augmentation, n_augmentations=1, data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = True, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__name: str = name
        self.__te_model: nn.Module = teacher_model
        self.__st_model: nn.Module = student_model
        self.__n_augmentations = n_augmentations
        self.__augmentation = augmentation
        self.__data_key = data_key
        self.__target_key = target_key

    def sample(self, k=1):
        samples = []
        sampled_rows = super().sample(k)
        for i in range(k):
            imgs = sampled_rows[i][self.__data_key]
            target = sampled_rows[i][self.__target_key]
            list_logits = []
            with torch.no_grad():
                te_logits = self.__te_model(imgs.to(next(self.__te_model.parameters()).device))
            list_imgs = []
            for b in range(imgs.shape[0]):
                for j in range(self.__n_augmentations):
                    img = imgs[b, :, :, :]
                    if img.shape[0] == 1:
                        img = img[0, :, :]
                    else:
                        img = img.permute(1, 2, 0)

                    img_cpu = to_cpu(img)
                    aug_img = self.__augmentation(img_cpu)
                    list_imgs.append(aug_img)

            batch_imgs = torch.stack(list_imgs, dim=0).to(next(self.__st_model.parameters()).device)
            logits = to_cpu(self.__st_model(batch_imgs))
            # del batch_imgs
            # gc.collect()
            list_logits.append(logits)
            if len(list_logits) > 1:
                st_logits = np.stack(list_logits, axis=1)
            elif len(list_logits) == 1:
                st_logits = list_logits[0]
            else:
                raise ValueError('Empty list!')
            # st_logits = torch.stack(list_logits, dim=0)
            samples.append({'name': self.__name, 'st_logits': st_logits, 'te_logits': te_logits, 'data': imgs, 'target': target})
        return samples


class FeatureMatchingSampler(ItemLoader):
    def __init__(self, model: nn.Module, latent_size: int, data_key: str = "data", meta_data: pd.DataFrame or None = None,
                 parse_item_cb: callable or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = True, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__model: nn.Module = model
        self.__latent_size: int = latent_size
        self.__data_key = data_key

    def sample(self, k=1):
        samples = []
        real_imgs_list = super().sample(k)
        for i in range(k):
            real_imgs = real_imgs_list[i][self.__data_key]
            features = self.__model.get_features(real_imgs)
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__model.parameters()).device)
            samples.append({'real_features': features.detach(), 'real_data': real_imgs, 'latent': noise_on_device})
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
    def __init__(self, g_network, batch_size, latent_size, n_classes, use_aux_target=False, same_class_batch=False):
        super().__init__(meta_data=None, parse_item_cb=None)
        self.__latent_size = latent_size
        self.batch_size = batch_size
        self.__g_network = g_network
        self.__n_classes = n_classes
        self.__use_aux_target = use_aux_target
        self.__sample_class_batch = same_class_batch

    def sample(self, k=1):
        samples = []
        for i in range(k):
            in_channels = self.__latent_size + self.__n_classes if self.__use_aux_target else self.__latent_size
            noise = torch.randn(self.batch_size, in_channels)
            noise_on_device = noise.to(next(self.__g_network.parameters()).device)

            fake: torch.Tensor = self.__g_network(noise_on_device)

            target = torch.zeros([self.batch_size, self.__n_classes + 1]).to(fake.device)

            if self.__use_aux_target:
                if self.__sample_class_batch:
                    target_cls = torch.ones([self.batch_size, 1], dtype=torch.int64)*(i%self.__n_classes)
                else:
                    target_cls = torch.randint(self.__n_classes, size=(self.batch_size, 1))
                target_val = torch.zeros([self.batch_size, 1], dtype=torch.int64)
                target_aux = torch.cat((target_cls, target_val), dim=1)

            samples.append({'data': fake.detach(),
                            'target': target,
                            'latent': noise,
                            'valid': target[:, -1],
                            'aux_target': target_aux if self.__use_aux_target else None})

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
            # target[:, -2] = 1.0
            samples.append({'latent': noise, 'target': target, 'valid': target[:, -1]})

        return samples

    def __len__(self):
        return 1