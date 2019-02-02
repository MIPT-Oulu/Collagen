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


class SSFoldSplit2(Splitter):
    def __init__(self, ds: pd.DataFrame, n_ss_folds: int = 3, n_folds: int = 5, target_col: str = 'target',
                 random_state: int or None = None, labeled_train_size: int = None, unlabeled_train_size: int = None, shuffle: bool = True):
        super().__init__()

        master_splitter = model_selection.StratifiedKFold(n_splits=n_ss_folds*n_folds, random_state=random_state)
        unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col]))
        unlabeled_ds = ds.iloc[unlabeled_idx]
        u_groups = ds[target_col].iloc[unlabeled_idx]
        labeled_ds = ds.iloc[labeled_idx]
        l_groups = ds[target_col].iloc[labeled_idx]

        if labeled_train_size > len(labeled_idx):
            raise ValueError('Input labeled train size {} is larger than actual labeled train size {}'.format(labeled_train_size, len(labeled_idx)))

        if unlabeled_train_size > len(unlabeled_idx):
            raise ValueError('Input unlabeled train size {} is larger than actual unlabeled train size {}'.format(unlabeled_train_size, len(unlabeled_idx)))

        self.__cv_folds_idx = []
        self.__ds_chunks = []
        unlabeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
        unlabeled_spl_iter = unlabeled_splitter.split(unlabeled_ds, unlabeled_ds[target_col], groups=u_groups)

        labeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
        labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col], groups=l_groups)

        for i in range(n_folds):
            u_train, u_test = next(unlabeled_spl_iter)
            l_train, l_test = next(labeled_spl_iter)
            if labeled_train_size is not None:
                chosen_l_train, _ = model_selection.train_test_split(l_train, train_size=labeled_train_size,
                                                                     random_state=random_state, shuffle=shuffle,
                                                                     stratify=labeled_ds[target_col])
            else:
                chosen_l_train = l_train
            if unlabeled_train_size is not None:
                chosen_u_train, _ = model_selection.train_test_split(u_train, train_size=unlabeled_train_size,
                                                                     random_state=random_state, shuffle=shuffle,
                                                                     stratify=unlabeled_ds[target_col])
            else:
                chosen_u_train = u_train
            self.__cv_folds_idx.append((chosen_l_train, l_test, chosen_u_train, u_test))
            self.__ds_chunks.append((labeled_ds.iloc[chosen_l_train], labeled_ds.iloc[l_test],
                                     unlabeled_ds.iloc[chosen_u_train], unlabeled_ds.iloc[u_test]))

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
