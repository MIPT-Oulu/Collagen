import dill as pickle
import pandas as pd
from sklearn import model_selection
from sklearn.utils import resample


class Splitter(object):
    def __init__(self):
        self.__ds_chunks = None
        self.__folds_iter = None
        pass

    def __next__(self):
        if self.__folds_iter is None:
            raise NotImplementedError
        else:
            next(self.__folds_iter)

    def __iter__(self):
        if self.__ds_chunks is None:
            raise NotImplementedError
        else:
            return self

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.__ds_chunks = pickle.load(f)
            self.__folds_iter = iter(self.__ds_chunks)


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
                 random_state: int or None = None, unlabeled_target_col: str = '5means_classes', test_ratio: int = 0.25,
                 labeled_train_size_per_class: int = None, unlabeled_train_size_per_class: int = None,
                 labeled_train_size: int = None, unlabeled_train_size: int = None, group_col: str or None = None,
                 equal_target: bool = True, equal_unlabeled_target: bool = True, shuffle: bool = True):
        super().__init__()

        self._test_ratio = test_ratio

        if equal_target and labeled_train_size_per_class is None:
            raise ValueError("labeled_train_size_per_class must be determined when \
            equal_target is True, but found None")

        # Master split into Label/Unlabel
        if group_col is None:
            master_splitter = model_selection.StratifiedKFold(n_splits=n_ss_folds, random_state=random_state)
            unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col]))
        else:
            master_splitter = model_selection.GroupKFold(n_splits=n_ss_folds)
            unlabeled_idx, labeled_idx = next(master_splitter.split(ds, ds[target_col], groups=ds[group_col]))
        unlabeled_ds = ds.iloc[unlabeled_idx]
        # u_groups = ds[unlabeled_target_col].iloc[unlabeled_idx]
        labeled_ds = ds.iloc[labeled_idx]
        l_groups = ds[target_col].iloc[labeled_idx]

        if not equal_target and labeled_train_size is not None and labeled_train_size > len(labeled_idx):
            raise ValueError('Input labeled train size {} is larger than actual labeled train size {}'.format(
                labeled_train_size, len(labeled_idx)))

        if unlabeled_train_size is not None and unlabeled_train_size > len(unlabeled_idx):
            unlabeled_train_size = len(unlabeled_idx)
            # raise ValueError('Input unlabeled train size {} is larger than actual unlabeled train size {}'.format(unlabeled_train_size, len(unlabeled_idx)))

        # Split labeled data using GroupKFold
        # Split unlabeled data using GroupKFold
        self.__cv_folds_idx = []
        self.__ds_chunks = []

        # split of train/val data
        if group_col is None:
            unlabeled_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state + 1)
            unlabeled_spl_iter = unlabeled_splitter.split(unlabeled_ds, unlabeled_ds[target_col])
        else:
            unlabeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
            unlabeled_spl_iter = unlabeled_splitter.split(unlabeled_ds, unlabeled_ds[target_col],
                                                          groups=unlabeled_ds[group_col])

        if group_col is None:
            labeled_splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state + 2)
            labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col])
        else:
            labeled_splitter = model_selection.GroupKFold(n_splits=n_folds)
            labeled_spl_iter = labeled_splitter.split(labeled_ds, labeled_ds[target_col], groups=labeled_ds[group_col])

        for i in range(n_folds):
            u_train, u_test = next(unlabeled_spl_iter)
            l_train, l_test = next(labeled_spl_iter)
            if equal_unlabeled_target:
                u_train_target = unlabeled_ds.iloc[u_train][unlabeled_target_col]
                u_test_target = unlabeled_ds.iloc[u_test][unlabeled_target_col]
            l_train_target = labeled_ds.iloc[l_train][target_col]
            l_train_data = labeled_ds.iloc[l_train]

            l_test_target = labeled_ds.iloc[l_test][target_col]
            l_test_data = labeled_ds.iloc[l_test]

            # Sample labeled_train_size of labeled data
            if equal_target:
                filtered_l_train_idx, chosen_l_train = self._sample_labeled_data(l_train_data, l_train_target,
                                                                                 target_col,
                                                                                 labeled_train_size_per_class,
                                                                                 random_state)

                filtered_l_test_idx, chosen_l_test = self._sample_labeled_data(l_test_data, l_test_target,
                                                                               target_col,
                                                                               int(
                                                                                   labeled_train_size_per_class * self._test_ratio),
                                                                               random_state)
            else:
                if labeled_train_size is not None:
                    chosen_l_train, _ = model_selection.train_test_split(l_train, train_size=labeled_train_size,
                                                                         random_state=random_state, shuffle=shuffle,
                                                                         stratify=l_train_target)
                    chosen_l_test, _ = model_selection.train_test_split(l_test, train_size=int(
                        labeled_train_size * self._test_ratio),
                                                                        random_state=random_state, shuffle=shuffle,
                                                                        stratify=l_train_target)
                else:
                    chosen_l_train = l_train
                    chosen_l_test = l_test
                filtered_l_train_idx = labeled_ds.iloc[chosen_l_train]
                filtered_l_test_idx = labeled_ds.iloc[chosen_l_test]

            # Sample unlabeled_train_size of labeled data
            if equal_unlabeled_target:
                filtered_u_train_idx, chosen_u_train = self._sample_unlabeled_data(unlabeled_ds, u_train,
                                                                                   unlabeled_target_col,
                                                                                   u_train_target,
                                                                                   unlabeled_train_size_per_class,
                                                                                   random_state)

                filtered_u_test_idx, chosen_u_test = self._sample_unlabeled_data(unlabeled_ds, u_test,
                                                                                 unlabeled_target_col,
                                                                                 u_test_target,
                                                                                 int(
                                                                                     unlabeled_train_size_per_class * self._test_ratio),
                                                                                 random_state)
            else:
                if unlabeled_train_size is not None:
                    # chosen_u_train, _ = model_selection.train_test_split(u_train, train_size=unlabeled_train_size,
                    #                                                      random_state=random_state, shuffle=shuffle)
                    is_replace = unlabeled_train_size > len(u_train)
                    chosen_u_train = resample(u_train, n_samples=unlabeled_train_size, replace=is_replace,
                                              random_state=random_state)
                    unlabeled_test_size = int(unlabeled_train_size * self._test_ratio)
                    is_replace = unlabeled_test_size > len(u_test)
                    chosen_u_test = resample(u_test, n_samples=unlabeled_test_size, replace=is_replace,
                                             random_state=random_state)
                else:
                    chosen_u_train = u_train
                    chosen_u_test = u_test

                filtered_u_train_idx = unlabeled_ds.iloc[chosen_u_train]
                filtered_u_test_idx = unlabeled_ds.iloc[chosen_u_test]

            self.__cv_folds_idx.append((chosen_l_train, chosen_l_test, chosen_u_train, chosen_u_test))

            self.__ds_chunks.append((filtered_l_train_idx, filtered_l_test_idx,
                                     filtered_u_train_idx, filtered_u_test_idx))

        self.__folds_iter = iter(self.__ds_chunks)

    def _sample_labeled_data(self, data, targets, target_col, data_per_class, random_state):
        labeled_targets = list(set(targets.tolist()))
        chosen_data = []
        for lt in labeled_targets:
            filtered_rows = data[data[target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            replace = data_per_class > len(filtered_rows_idx)
            chosen_idx_by_target = resample(filtered_rows_idx, n_samples=data_per_class,
                                            replace=replace, random_state=random_state)
            chosen_data += chosen_idx_by_target.tolist()
        filtered_idx = data.loc[chosen_data]
        return filtered_idx, chosen_data

    def _sample_unlabeled_data(self, unlabeled_ds, u_train, unlabeled_target_col, u_train_target, data_per_class,
                               random_state, replace=False):
        u_train_target = unlabeled_ds.iloc[u_train][unlabeled_target_col]
        u_train_data = unlabeled_ds.iloc[u_train]
        ideal_labeled_targets = list(set(u_train_target.tolist()))
        chosen_u_train = []
        for lt in ideal_labeled_targets:
            filtered_rows = u_train_data[u_train_data[unlabeled_target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            replace = data_per_class > len(filtered_rows_idx)
            chosen_u_train_by_target = resample(filtered_rows_idx, n_samples=data_per_class,
                                                replace=replace, random_state=random_state)
            chosen_u_train += chosen_u_train_by_target.tolist()
        filtered_u_train_idx = u_train_data.loc[chosen_u_train]
        return filtered_u_train_idx, chosen_u_train

    def _sampling(self, l_train_data, l_train_target, target_col, labeled_train_size_per_class, random_state):
        labeled_targets = list(set(l_train_target.tolist()))
        chosen_l_train = []
        for lt in labeled_targets:
            filtered_rows = l_train_data[l_train_data[target_col] == lt]
            filtered_rows_idx = filtered_rows.index
            chosen_l_train_by_target = resample(filtered_rows_idx, n_samples=labeled_train_size_per_class, replace=True,
                                                random_state=random_state)
            chosen_l_train += chosen_l_train_by_target.tolist()
        filtered_l_train_idx = l_train_data.loc[chosen_l_train]
        return chosen_l_train, filtered_l_train_idx

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f)

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
