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
        super().__init__()
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


class SSFoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, n_folds: int = 10, target_col: str = 'target',
                 group_col: str or None = None, random_state: int or None = None,
                 unlabeled_size: float = 0.5, val_size: float = 0.4, shuffle: bool = True):
        super().__init__()

        if unlabeled_size + val_size + 1.0 / n_folds > 1:
            raise ValueError("Requested data exceed 100% ({}+{}+{})".format(unlabeled_size, val_size + 1.0 / n_folds))

        if group_col is None:
            splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state)
            split_iter = splitter.split(ds, ds[target_col])
        else:
            splitter = model_selection.GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(ds, ds[target_col], groups=ds[group_col])

        self.__cv_folds_idx = []
        for (other1_idx, train_labeled_idx) in split_iter:
            val_idx, other2_idx = model_selection.train_test_split(other1_idx,
                                                                   train_size=val_size,
                                                                   shuffle=shuffle,
                                                                   random_state=random_state)
            if unlabeled_size + val_size + 1.0 / n_folds < 1 - 1e-3:
                train_unlabeled_idx, _ = model_selection.train_test_split(other2_idx,
                                                                          train_size=val_size,
                                                                          shuffle=shuffle,
                                                                          random_state=random_state)
            else:
                train_unlabeled_idx = other2_idx
            self.__cv_folds_idx.append((train_labeled_idx, train_unlabeled_idx, val_idx))

        self.__ds_chunks = [(ds.iloc[split[0]], ds.iloc[split[1]], ds.iloc[split[2]]) for split in self.__cv_folds_idx]
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


class SSFoldSplit2(Splitter):
    def __init__(self, ds: pd.DataFrame, n_ss_folds: int = 3, n_folds: int = 5, target_col: str = 'target',
                 group_col: str or None = None, random_state: int or None = None,
                 unlabeled_size: float = 0.5, shuffle: bool = True):
        super().__init__()

        master_splitter = model_selection.StratifiedKFold(n_splits=n_ss_folds*n_folds, random_state=random_state)
        unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col]))
        unlabeled_ds = ds.iloc[unlabeled_idx]
        u_groups = ds[target_col].iloc[unlabeled_idx]
        labeled_ds = ds.iloc[labeled_idx]
        l_groups = ds[target_col].iloc[labeled_idx]

        self.__cv_folds_idx = []

        unlabeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
        unlabeled_spl_iter = unlabeled_splitter.split(unlabeled_ds, unlabeled_ds[target_col], groups=u_groups)

        labeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
        labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col], groups=l_groups)

        for i in range(n_folds):
            u_train, u_test = next(unlabeled_spl_iter)
            l_train, l_test = next(labeled_spl_iter)
            self.__cv_folds_idx.append((l_train, l_test, u_train, u_test))

        self.__ds_chunks = [(labeled_ds.iloc[split[0]], labeled_ds.iloc[split[1]], unlabeled_ds.iloc[split[2]], unlabeled_ds.iloc) for split in self.__cv_folds_idx]
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


class TrainValSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, train_size: int or float, shuffle: bool, random_state: int or None = None):
        super().__init__()
        train_idx, val_idx = model_selection.train_test_split(self.__ds_chunks.index,
                                                              train_size=train_size,
                                                              shuffle=shuffle,
                                                              random_state=random_state)
        self.__ds_chunks = [ds.iloc[train_idx], ds.iloc[val_idx]]
        self.__dataset_iter = iter(self.__ds_chunks)

    def __next__(self):
        return next(self.__dataset_iter)

    def __iter__(self):
        return self

    def train_set(self):
        return self.__ds_chunks[0]

    def val_set(self):
        return self.__ds_chunks[1]
