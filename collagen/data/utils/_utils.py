import copy
from sklearn import model_selection
import pandas as pd


class ApplyTransform(object):
    """Applies a callable transform to certain objects in iterable using given indices.

    Parameters
    ----------
    transform: callable
        Callable transform to be applied
    idx: int or tuple or or list None
        Index or set of indices, where the transform will be applied.

    """
    def __init__(self, transform: callable, idx: int  or tuple or list = 0):
        self.__transform: callable = transform
        if isinstance(idx, int):
            idx = (idx, )
        self.__idx: int or tuple or list = idx

    def __call__(self, items):
        """
        Applies a transform to the given sequence of elements.

        Uses the locations (indices) specified in the constructor.

        Parameters
        ----------
        items: tuple or list
            Set of items
        Returns
        -------
        result: tuple
            Transformed list

        """

        if self.__idx is None:
            return items

        if not isinstance(items, (tuple, list)):
            raise TypeError

        idx = set(self.__idx)
        res = []
        for i, item in enumerate(items):
            if i in idx:
                res.append(self.__transform(item))
            else:
                res.append(copy.deepcopy(item))

        return tuple(res)


class Normalize(object):
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, tensor, inplace=True):
        if not inplace:
            tensor_trf = tensor.copy()
        else:
            tensor_trf = tensor

        if len(tensor_trf.size()) != 3:
            raise ValueError(f'Input tensor must have 3 dimensions (CxHxW), but found {len(tensor_trf)}')

        if tensor_trf.size(0) != len(self.__mean):
            raise ValueError(f'Incompatible number of channels. '
                             f'Mean has {len(self.__mean)} channels, tensor - {tensor_trf.size()}')

        if tensor_trf.size(0) != len(self.__std):
            raise ValueError(f'Incompatible number of channels. '
                             f'Std has {len(self.__mean)} channels, tensor - {tensor_trf.size()}')

        for channel in range(tensor_trf.size(0)):
            tensor_trf[channel, :, :] -= self.__mean[channel]
            tensor_trf[channel, :, :] /= self.__std[channel]

        return tensor_trf


class Compose(object):
    def __init__(self, transforms: list or tuple):
        self.__transforms = transforms

    def __call__(self, x):
        for trf in self.__transforms:
            x = trf(x)

        return x


class FoldSplit(object):
    def __init__(self, ds: pd.DataFrame, n_folds: int = 5, target_col: str = 'target',
                 group_col: str or None = None, random_state: int or None = None):

        if group_col is None:
            splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state)
            split_iter = splitter.split(ds, ds[target_col])
        else:
            splitter = model_selection.GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(ds, ds[target_col], groups=ds[group_col])

        self.__cv_folds_idx = [(train_idx, val_idx) for (train_idx, val_idx) in split_iter]
        self.__ds_chunks = [(ds.iloc[split[0]], ds.iloc[split[1]]) for split in self.__cv_folds_idx]
        self.__folds_iter = iter(self.__ds_chunks)

    def __next__(self):
        return next(self.__folds_iter)

    def __iter__(self):
        return self

    def fold(self, i):
        return self.__ds_chunks[i]

    def n_folds(self):
        return len(self.__cv_folds_idx)

    def fold_idx(self, i):
        return self.__cv_folds_idx


