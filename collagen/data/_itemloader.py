import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
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
        will be smaller. (the default is False)
    timeout : int, optional
        If positive, the timeout value for collecting a batch from workers.
        If ``0``, ignores ``timeout`` notion. Must be non-negative. (the default is 0)
    """

    def __init__(self, meta_data: pd.DataFrame or None = None, parse_item_cb: callable or None = None,
                 root: str or None = None, batch_size: int = 1,
                 num_workers: int = 0, shuffle: bool = False, pin_memory: bool = False,
                 collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None,
                 batch_sampler=None, drop_last: bool = False, timeout: int = 0, name:str = "loader"):
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

            batch['name'] = self.__name
            samples.append(batch)

        return samples


class MixUpSampler(ItemLoader):
    def __init__(self, name: str, alpha: callable or float, model: nn.Module or None = None,
                 data_rearrange: callable or None = None, target_rearrage: callable or None = None,
                 data_key: str = "data", target_key: str = 'target', min_lambda: float = 0.7,
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None, compute_logits_mixup:bool = True,
                 drop_last: bool = False, timeout: int = 0, detach: bool = False):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)

        self.__model = model
        self.__name = name
        self.__data_key = data_key
        self.__target_key = target_key
        self.__alpha = alpha
        self.__data_rearrange = self._default_change_data_ordering if data_rearrange is None else data_rearrange
        self.__min_l = min_lambda
        self.__compute_logits_mixup = compute_logits_mixup

    @property
    def alpha(self):
        return self.__alpha

    @staticmethod
    def _default_change_data_ordering(x, y):
        return torch.flip(x, dims=[0]), torch.flip(y, dims=[0])

    def __len__(self):
        return super().__len__()

    def sample(self, k=1):
        sampled_rows = super().sample(k)
        device = next(self.__model.parameters()).device
        samples = []
        for i in range(k):
            imgs1 = sampled_rows[i][self.__data_key]
            target1 = sampled_rows[i][self.__target_key]

            imgs2, target2 = self.__data_rearrange(imgs1, target1)

            if callable(self.alpha):
                l = self.alpha()
            elif isinstance(self.alpha, float):
                l = self.alpha
            else:
                raise ValueError('Not support alpha of {}'.format(type(self.alpha)))

            if not isinstance(l, float) or l < 0 or l > 1:
                raise ValueError('Alpha {} is not float or out of range [0,1]'.format(l))
            elif l < 0.5:
                l = 1 - l

            l = max(l, self.__min_l)

            mixup_imgs = l * imgs1 + (1 - l) * imgs2

            logits1 = self.__model(imgs1.to(device))
            logits2 = self.__model(imgs2.to(device))

            if self.__compute_logits_mixup:
                logits_mixup = l * logits1 + (1 - l) * logits2
            else:
                logits_mixup = None

            samples.append({'name': self.__name, 'mixup_data': mixup_imgs, 'target': target1, 'target_bg': target2,
                            'alpha': l, 'logits_mixup': logits_mixup})
        return samples


class MixUpSampler2(ItemLoader):
    def __init__(self, name: str, alpha: callable or float, model: nn.Module or None, 
                 data_rearrange: callable or None = None, target_rearrage: callable or None = None,
                 data_key: str = "data", target_key: str = 'target', augmentation: callable or None = None,
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None, min_lambda: float = 0.7,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0, detach: bool = False):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)

        self.__model = model
        self.__name = name
        # self.__n_classes = n_classes
        self.__data_key = data_key
        self.__target_key = target_key
        self.__alpha = alpha
        self.__data_rearrange = self._default_change_data_ordering if data_rearrange is None else data_rearrange

        self.__augmentation = augmentation
        self.__min_l = min_lambda

    @property
    def alpha(self):
        return self.__alpha

    @staticmethod
    def _default_change_data_ordering(x, y):
        return torch.flip(x, dims=[0]), torch.flip(y, dims=[0])

    def __len__(self):
        return super().__len__()

    def sample(self, k=1):
        sampled_rows = super().sample(k)
        device = next(self.__model.parameters()).device
        samples = []
        for i in range(k):
            imgs1 = sampled_rows[i][self.__data_key]
            target1 = sampled_rows[i][self.__target_key]

            imgs2, target2 = self.__data_rearrange(imgs1, target1)

            if callable(self.alpha):
                l = self.alpha()
            elif isinstance(self.alpha, float):
                l = self.alpha
            else:
                raise ValueError('Not support alpha of {}'.format(type(self.alpha)))

            if not isinstance(l, float) or l < 0 or l > 1:
                raise ValueError('Alpha {} is not float or out of range [0,1]'.format(l))
            elif l < 0.5:
                l = 1 - l

            l = max(l, self.__min_l)

            mixup_imgs = l * imgs1 + (1 - l) * imgs2

            logits1 = self.__model(imgs1.to(device))
            logits2 = self.__model(imgs2.to(device))
            # mixup_logits = self.__model(mixup_imgs.to(device))

            imgs1_cpu = to_cpu(imgs1.permute(0, 2, 3, 1), use_numpy=True)
            imgs1_aug = self.__augmentation(imgs1_cpu)
            logits1_aug = self.__model(imgs1_aug.to(device))

            logits_mixup = l*logits1 + (1 - l)*logits2
            samples.append({'name': self.__name, 'mixup_data': mixup_imgs,
                            'target': target1, 'target_bg': target2,
                            'logits': logits1, 'logits_aug': logits1_aug,
                            'logits_mixup': logits_mixup, 'alpha': l})
        return samples


class AugmentedGroupSampler(ItemLoader):
    def __init__(self, model: nn.Module, name: str, augmentation, n_augmentations=1, output_type='logits', data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0, detach: bool = False):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__name = name
        self.__model: nn.Module = model
        self.__n_augmentations = n_augmentations
        self.__augmentation = augmentation
        self.__data_key = data_key
        self.__target_key = target_key
        self.__output_type = output_type
        self.__detach = detach

    def __len__(self):
        return super().__len__()

    def sample(self, k=1):
        samples = []
        sampled_rows = super().sample(k)
        for i in range(k):
            imgs = sampled_rows[i][self.__data_key]
            target = sampled_rows[i][self.__target_key]
            batch_size = imgs.shape[0]
            list_logits = []

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

            batch_imgs = torch.stack(list_imgs, dim=0)
            batch_imgs = batch_imgs.to(next(self.__model.parameters()).device)
            if self.__output_type == 'logits':
                out = self.__model(batch_imgs)
                # logits = to_cpu(out, use_numpy=False, required_grad=True)
                logits = out
            elif self.__output_type == 'features':
                out = self.__model.get_features(batch_imgs)
                # logits = to_cpu(out, use_numpy=False, required_grad=True)
                logits = out

            if self.__n_augmentations > 1:
                logits = logits.view(self.__n_augmentations, batch_size, -1)
            elif self.__n_augmentations == 1:
                pass
            else:
                raise ValueError('Empty list!')

            if self.__detach:
                logits = logits.detach()

            samples.append({'name': self.__name, 'logits': logits , 'data': imgs, 'target': target})
        return samples


class AugmentedGroupSampler2(ItemLoader):
    def __init__(self, model: nn.Module, te_model: nn.Module, name: str, augmentation, n_augmentations=1, output_type='logits', data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None, meta_data: pd.DataFrame or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0, detach: bool = False):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__name = name
        self.__model: nn.Module = model
        self.__te_model: nn.Module = te_model
        self.__n_augmentations = n_augmentations
        self.__augmentation = augmentation
        self.__data_key = data_key
        self.__target_key = target_key
        self.__output_type = output_type
        self.__detach = detach

    def __len__(self):
        return super().__len__()

    def sample(self, k=1):
        samples = []
        sampled_rows = super().sample(k)
        for i in range(k):
            imgs = sampled_rows[i][self.__data_key]
            target = sampled_rows[i][self.__target_key]
            batch_size = imgs.shape[0]
            list_logits = []

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

            batch_imgs = torch.stack(list_imgs, dim=0)
            batch_imgs = batch_imgs.to(next(self.__model.parameters()).device)

            te_f = self.__te_model.get_features(batch_imgs)

            st_f = self.__model.get_features(batch_imgs.requires_grad_(True))
            # logits = self.__model(batch_imgs)
            # f = self.__model.get_features(batch_imgs)
            # logits = self.__model.forward_features(f)

            if self.__n_augmentations > 1:
                # logits = logits.view(self.__n_augmentations, batch_size, -1)
                te_f = te_f.view(self.__n_augmentations, batch_size, -1)
                st_f = st_f.view(self.__n_augmentations, batch_size, -1)
            elif self.__n_augmentations == 1:
                pass
            else:
                raise ValueError('Empty list!')

            te_f = te_f.detach()

            samples.append({'name': self.__name, 'st_features': st_f, 'te_features': te_f, 'data': imgs, 'target': target})
        return samples


class FeatureMatchingSampler(ItemLoader):
    def __init__(self, model: nn.Module, latent_size: int, data_key: str = "data", meta_data: pd.DataFrame or None = None,
                 parse_item_cb: callable or None = None, name='fm',
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0):
        super().__init__(meta_data=meta_data, parse_item_cb=parse_item_cb, root=root, batch_size=batch_size,
                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, collate_fn=collate_fn,
                         transform=transform, sampler=sampler, batch_sampler=batch_sampler, drop_last=drop_last, timeout=timeout)
        self.__model: nn.Module = model
        self.__latent_size: int = latent_size
        self.__data_key = data_key
        self.__name = name

    def sample(self, k=1):
        samples = []
        real_imgs_list = super().sample(k)
        for i in range(k):
            real_imgs = real_imgs_list[i][self.__data_key]
            features = self.__model.get_features(real_imgs)
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__model.parameters()).device)
            samples.append({'name': self.__name, 'real_features': features.detach(), 'real_data': real_imgs, 'latent': noise_on_device})
        return samples


class GANFakeSampler(ItemLoader):
    def __init__(self, g_network, batch_size, latent_size, name='ganfake'):
        super().__init__(meta_data=None, parse_item_cb=None, name=name)
        self.__latent_size = latent_size
        self.batch_size = batch_size
        self.__g_network = g_network
        self.__name = name

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, self.__latent_size)
            noise_on_device = noise.to(next(self.__g_network.parameters()).device)
            fake: torch.Tensor = self.__g_network(noise_on_device)
            samples.append({'name': self.__name, 'data': fake.detach(), 'target': torch.zeros(self.batch_size).to(fake.device), 'latent': noise})

        return samples

    def __len__(self):
        return 1


class SSGANFakeSampler(ItemLoader):
    def __init__(self, g_network, batch_size, latent_size, n_classes, name='ssgan_fake', use_aux_target=False, same_class_batch=False):
        super().__init__(meta_data=None, parse_item_cb=None, name=name)
        self.__latent_size = latent_size
        self.batch_size = batch_size
        self.__g_network = g_network
        self.__n_classes = n_classes
        self.__use_aux_target = use_aux_target
        self.__sample_class_batch = same_class_batch
        self.__name = name

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

            samples.append({'name': self.__name,
                            'data': fake.detach(),
                            'target': target,
                            'onehot': target[:, :-1],
                            'latent': noise,
                            'valid': target[:, -1],
                            'aux_target': target_aux if self.__use_aux_target else None})

        return samples

    def __len__(self):
        return 1


class GaussianNoiseSampler(ItemLoader):
    def __init__(self, batch_size, latent_size, device, n_classes, name='gaussian_noise'):
        super().__init__(meta_data=None, parse_item_cb=None, name=name)
        self.latent_size = latent_size
        self.device = device
        self.batch_size = batch_size
        self.__n_classes = n_classes
        self.__name = name

    def sample(self, k=1):
        samples = []
        for _ in range(k):
            noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
            target = torch.zeros([self.batch_size, self.__n_classes + 1]).to(self.device)
            # target[:, -2] = 1.0
            samples.append({'name': self.__name, 'latent': noise, 'target': target, 'valid': target[:, -1]})

        return samples

    def __len__(self):
        return 1