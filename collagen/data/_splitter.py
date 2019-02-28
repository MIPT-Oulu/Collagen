from sklearn import model_selection
from sklearn.utils import resample
from collagen.data import ItemLoader
from random import shuffle as shuffle_list
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
    def __init__(self, ds: pd.DataFrame, n_ss_folds: int = 3, n_folds: int = 5, target_col: str = 'target',
                 random_state: int or None = None, unlabeled_target_col: str = '5means_classes',
                 labeled_train_size_per_class: int = None, unlabeled_train_size_per_class: int = None,
                 labeled_train_size: int = None, unlabeled_train_size: int = None,
                 equal_target: bool = True, equal_unlabeled_target: bool = True, shuffle: bool = True):
        super().__init__()
        if equal_target and labeled_train_size_per_class is None:
            raise ValueError("labeled_train_size_per_class must be determined when equal_target is True, but found None")
        # Master split into Label/Unlabel
        master_splitter = model_selection.StratifiedKFold(n_splits=n_ss_folds, random_state=random_state)
        unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col]))
        unlabeled_ds = ds.iloc[unlabeled_idx]
        u_groups = ds[target_col].iloc[unlabeled_idx]
        labeled_ds = ds.iloc[labeled_idx]
        l_groups = ds[target_col].iloc[labeled_idx]

        if not equal_target and labeled_train_size is not None and labeled_train_size > len(labeled_idx):
            raise ValueError('Input labeled train size {} is larger than actual labeled train size {}'.format(labeled_train_size, len(labeled_idx)))

        if unlabeled_train_size is not None and unlabeled_train_size > len(unlabeled_idx):
            raise ValueError('Input unlabeled train size {} is larger than actual unlabeled train size {}'.format(unlabeled_train_size, len(unlabeled_idx)))

        # Split labeled data using GroupKFold
        # Split unlabeled data using GroupKFold
        self.__cv_folds_idx = []
        self.__ds_chunks = []

        # split of unlabeled data
        unlabeled_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state+1)
        unlabeled_spl_iter = unlabeled_splitter.split(unlabeled_ds, unlabeled_ds[target_col])

        labeled_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state+2)
        labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col])

        for i in range(n_folds):
            u_train, u_test = next(unlabeled_spl_iter)
            l_train, l_test = next(labeled_spl_iter)
            u_train_target = unlabeled_ds.iloc[u_train][target_col]
            l_train_target = labeled_ds.iloc[l_train][target_col]
            l_train_data = labeled_ds.iloc[l_train]

            # Sample labeled_train_size of labeled data
            if equal_target:
                labeled_targets = list(set(l_train_target.tolist()))
                chosen_l_train = []
                for lt in labeled_targets:
                    filtered_rows = l_train_data[l_train_data[target_col] == lt]
                    filtered_rows_idx = filtered_rows.index
                    chosen_l_train_by_target = resample(filtered_rows_idx, n_samples=labeled_train_size_per_class, replace=True, random_state=random_state)
                    chosen_l_train += chosen_l_train_by_target.tolist()
                filtered_l_train_idx = l_train_data.loc[chosen_l_train]
            else:
                if labeled_train_size is not None:
                    chosen_l_train, _ = model_selection.train_test_split(l_train, train_size=labeled_train_size,
                                                                         random_state=random_state, shuffle=shuffle,
                                                                         stratify=l_train_target)
                else:
                    chosen_l_train = l_train
                filtered_l_train_idx = labeled_ds.iloc[chosen_l_train]
            # Sample unlabeled_train_size of labeled data
            # TODO: Use clustering instead
            if equal_unlabeled_target:
                u_train_target = unlabeled_ds.iloc[u_train][target_col]
                u_train_data = unlabeled_ds.iloc[u_train]
                ideal_labeled_targets = list(set(u_train_target.tolist()))
                chosen_u_train = []
                for lt in ideal_labeled_targets:
                    filtered_rows = u_train_data[u_train_data[target_col] == lt]
                    filtered_rows_idx = filtered_rows.index
                    chosen_u_train_by_target = resample(filtered_rows_idx, n_samples=unlabeled_train_size_per_class,
                                                        replace=True, random_state=random_state)
                    chosen_u_train += chosen_u_train_by_target.tolist()
                filtered_u_train_idx = u_train_data.loc[chosen_u_train]
            else:
                if unlabeled_train_size is not None:
                    chosen_u_train, _ = model_selection.train_test_split(u_train, train_size=unlabeled_train_size,
                                                                         random_state=random_state, shuffle=shuffle,
                                                                         stratify=u_train_target)
                else:
                    chosen_u_train = u_train
                filtered_u_train_idx = unlabeled_ds.iloc[chosen_u_train]

            self.__cv_folds_idx.append((chosen_l_train, l_test, chosen_u_train, u_test))


            self.__ds_chunks.append((filtered_l_train_idx,   labeled_ds.iloc[l_test],
                                     filtered_u_train_idx, unlabeled_ds.iloc[u_test]))

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
