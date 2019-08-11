import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import pandas as pd
import numpy as np

from collagen.data import ItemLoader
from collagen.core.utils import to_cpu


class MixMatchSampler(object):
    def __init__(self, model: nn.Module, name: str, augmentation, labeled_meta_data: pd.DataFrame,
                 unlabeled_meta_data: pd.DataFrame,
                 n_augmentations=1, output_type='logits', data_key: str = "data", target_key: str = 'target',
                 parse_item_cb: callable or None = None,
                 root: str or None = None, batch_size: int = 1, num_workers: int = 0, shuffle: bool = False,
                 pin_memory: bool = False, collate_fn: callable = default_collate, transform: callable or None = None,
                 sampler: torch.utils.data.sampler.Sampler or None = None, batch_sampler=None,
                 drop_last: bool = False, timeout: int = 0, detach: bool = False):
        self._label_sampler = ItemLoader(meta_data=labeled_meta_data, parse_item_cb=parse_item_cb, root=root,
                                         batch_size=batch_size,
                                         num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                                         collate_fn=collate_fn,
                                         transform=transform, sampler=sampler, batch_sampler=batch_sampler,
                                         drop_last=drop_last, timeout=timeout)

        self._unlabel_sampler = ItemLoader(meta_data=unlabeled_meta_data,
                                           parse_item_cb=parse_item_cb, root=root,
                                           batch_size=batch_size,
                                           num_workers=num_workers, shuffle=shuffle,
                                           pin_memory=pin_memory, collate_fn=collate_fn,
                                           transform=transform, sampler=sampler,
                                           batch_sampler=batch_sampler, drop_last=drop_last,
                                           timeout=timeout)

        self._name = name
        self._model: nn.Module = model
        self._n_augmentations = n_augmentations
        self._augmentation = augmentation
        self._data_key = data_key
        self._target_key = target_key
        self._output_type = output_type
        self._detach = detach
        self._len = max(len(self._label_sampler, self._unlabel_sampler))

    def __len__(self):
        return self._len

    def _crop_if_needed(self, df1, df2):
        min_len = min(len(df1), len(df2))
        df1 = df1[:min_len]
        df2 = df2[:min_len]
        return df1, df2

    def sharpen(self, x, T=0.5):
        assert len(x.shape) == 2

        _x = torch.pow(x, T)
        s = torch.sum(_x, dim=-1, keepdim=True)
        _x = _x / s
        return _x

    def sample(self, k=1):
        samples = []
        labeled_sampled_rows = self._label_sampler.sample(k)
        unlabeled_sampled_rows = self._unlabel_sampler.sample(k)

        labeled_sampled_rows, unlabeled_sampled_rows = self._crop_if_needed(labeled_sampled_rows, unlabeled_sampled_rows)

        for i in range(k):
            u_imgs = unlabeled_sampled_rows[i][self._data_key]

            list_imgs = []
            for b in range(u_imgs.shape[0]):
                for j in range(self._n_augmentations):
                    img = u_imgs[b, :, :, :]
                    if img.shape[0] == 1:
                        img = img[0, :, :]
                    else:
                        img = img.permute(1, 2, 0)

                    img_cpu = to_cpu(img)
                    aug_img = self._augmentation(img_cpu)
                    list_imgs.append(aug_img)

            batch_imgs = torch.stack(list_imgs, dim=0)
            batch_imgs = batch_imgs.to(next(self._model.parameters()).device)
            if self._output_type == 'logits':
                out = self._model(batch_imgs)
                # logits = to_cpu(out, use_numpy=False, required_grad=True)
                logits = out
            elif self._output_type == 'features':
                out = self._model.get_features(batch_imgs)
                # logits = to_cpu(out, use_numpy=False, required_grad=True)
                logits = out

            preds = out.view(u_imgs.shape[0], -1, out.shape[-1])
            mean_preds = torch.mean(preds, dim=1)
            guessing_labels = self.sharpen(mean_preds).detach()

            unlabeled_sampled_rows[i][self._target_key] = guessing_labels

        union_rows = labeled_sampled_rows + unlabeled_sampled_rows

        rand_idx = np.random.permutation(len(union_rows))

        for i in range(k):
            u_imgs = unlabeled_sampled_rows[i][self._data_key]


            if self._detach:
                logits = logits.detach()

            samples.append({'name': self._name, 'logits': logits, 'data': imgs, 'target': target})
        return samples
