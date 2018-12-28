import copy


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
        self.__transfroms = transforms

    def __call__(self, x):
        for trf in self.__transfroms:
            x = trf(x)

        return x

class StratifiedShuffleSplit(object):
    def __init__(self, n_folds, random_state, ):
        sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=seed)
        cv_folds = [(train_idx, val_idx) for (train_idx, val_idx) in sss.split(train_ds, train_ds.target)]


