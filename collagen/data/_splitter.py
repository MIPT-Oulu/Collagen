from sklearn import model_selection
from collagen.data import ItemLoader
import pandas as pd


class Splitter(object):
    def __init__(self):
        pass

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class FoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, n_folds: int = 5, target_col: str = 'target',
                 group_col: str or None = None, random_state: int or None = None):
        super(FoldSplit, self).__init__()
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
