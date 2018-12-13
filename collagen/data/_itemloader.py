import sys
import os
import numpy as numpy
import pandas as pd
import torchvision
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset

class _itemloader(Dataset):
    r"""
    Item loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset. Presume to use 
        `torchvision.datasets.ImageFolder` and `torch.utils.data.DataLoader`
        to create `data_loader`

    Arguments in `config_dict`:
        data_root: root dir to load the data.
        meta_dataframe: Pandas dataframe of data and target. 
            Column name of class is `class`, and it's required column.
        loader: function to load single item/datum
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)

    .. note:: By default, each worker will have its PyTorch seed set to
              ``base_seed + worker_id``, where ``base_seed`` is a long generated
              by main process using its RNG. However, seeds for other libraies
              may be duplicated upon initializing workers (w.g., NumPy), causing
              each worker to return identical random numbers. (See
              :ref:`dataloader-workers-random-seed` section in FAQ.) You may
              use :func:`torch.initial_seed()` to access the PyTorch seed for
              each worker in :attr:`worker_init_fn`, and use it to set other
              seeds before data loading.
    """
    def __init__(self, config_dict):

        self.__initialized = False
        self.required_args = ["data_root", "meta_dataframe", "loader"]
        self.optional_args = ["batch_size", "num_threads", "shuffle", "pin_memory", "collate_fn", "transform", 
                                "target_transform", "sampler", "batch_sampler", "drop_last", "timeout", "worker_init_fn"]
        
        # Setup default values of `config_dict`
        self.config_dict = {"batch_size": 1, "num_workers": 1, "shuffle": True}
        self.config_dict["callback"] = None
        self.config_dict["pin_memory"] = False
        self.config_dict["collate_fn"] = None
        self.config_dict["transform"] = None
        self.config_dict["target_transform"] = None
        self.config_dict["batch_sampler"] = None
        self.config_dict["sampler"] = None
        self.config_dict["drop_last"] = None
        self.config_dict["timeout"] = None
        self.config_dict["worker_init_fn"] = None
        
        # File extension
        self.extensions = ['.png', '.jpg']

        # Setup `config_dict`
        if config_dict is not None:
            self._setup(config_dict)

        # Make dataset
        samples = self._make_dataset(self.config_dict["data_root"], self.config_dict["meta_dataframe"], self.extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.config_dict["data_root"] + "\n"
                               "Supported extensions are: " + ",".join(self.extensions)))

        ##### ----- Required internal vars ----- #####
        self.target = []
        for s in samples:
            self.target.append(s[1])
        
        all_classes = []

        if isinstance(self.target[0], dict):
            if "class" in self.target[0]:
                for t in self.target:
                    all_classes.append(t["class"])
            else:
                raise ValueError("Dict target must include `class` column, but only found {}".format(self.target[0]))
        elif isinstance(self.target[0], str) or isinstance(self.target[0], int):
            for t in self.target:
                all_classes.append(t)
        else:
            raise ValueError("Not support target type {}".format(type(self.target[0])))

        self.classes = list(set(all_classes.sort()))
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        self.class_to_idx = class_to_idx
        self.root = self.config_dict["data_root"]
        self.loader = self.config_dict["loader"]    # Load simple item
        self.transform = self.config_dict["transform"]
        self.target_transform = self.config_dict["target_transform"]

        # Assume to use `torch.utils.data.DataLoader`
        self._data = torchvision.datasets.ImageFolder(self.root, transform=self.transform, target_transform=self.target_transform)
        self.data_loader = torch.utils.data.DataLoader(self._data,
                                                        batch_size=self.config_dict["batch_size"],
                                                        shuffle=self.config_dict["shuffle"],
                                                        sampler=self.config_dict["sampler"],
                                                        batch_sampler=self.config_dict["batch_sampler"],
                                                        num_workers=self.config_dict["num_workers"],
                                                        collate_fn=self.config_dict["collate_fn"],
                                                        pin_memory=self.config_dict["pin_memory"],
                                                        drop_last=self.config_dict["drop_last"],
                                                        timeout=self.config_dict["timeout"],
                                                        worker_init_fn=self.config_dict["worker_init_fn"])
        ##### ----- Required internal vars ----- #####

    def _has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

    def _make_dataset(self, root, df, extension, split_value=None, split_name=None, list_target_column_names=["class"]):
        images = []
        root = os.path.expanduser(root)
        if split_name is not None and  split_name in df.columns:
            _df = df[df[split_name] == split_value]
        else:
            _df = df

        for index, row in _df.iterrows():
            fname = row["file_name"]
            if self._has_file_allowed_extension(fname, extension):
                path = os.path.join(root, fname)
                target = dict()
                for col_name in list_target_column_names:
                    if col_name in _df.columns:
                        target[col_name] = row[col_name]
                if len(target) > 1:
                    pass
                elif len(target) == 1:
                    for k in target:
                        target = target[k]
                else:
                    raise ValueError("There must be at least 1 label, but found 0")
                
                item = (path, target)
                images.append(item)

        return images

    def _check_param_exists(self, param_name):
        return param_name in self.config_dict

    def _check_exists(self):
        return len(self.config_dict["meta_dataframe"]) < 1

    def _setup(self, config_dict):
        # Validate input
        # Check required parameters
        if not isinstance(config_dict, dict):
            raise ValueError("Input config must be `dict`, but found {}".format(type(config_dict)))
        
        for arg in self.required_args:
            if arg not in config_dict:
                raise ValueError("Input config must include `{}`".format(arg))

        for k in config_dict:
            self.config_dict[k] = config_dict[k]

        if not os.path.exists(self.config_dict["data_root"]):
            raise ValueError("Not found data_root {}".format(self.config_dict["data_root"]))

        if not self._check_exists():
            raise ValueError("Input dataframe is empty")

        # Check optional parameters
        for arg in self.optional_args:
            if arg in config_dict and arg == "batch_size":
                self.config_dict[arg] = max(1, config_dict[arg])
            elif arg in config_dict and arg == "num_workers":
                self.config_dict[arg] = max(1, config_dict[arg])

        # Assumption: Clearing invalid rows should have been done already
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

    def __len__(self):
        return len(self.config_dict["meta_dataframe"])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.config_dict["transform"].__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.config_dict["target_transform"].__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.config_dict["samples"][index]
        sample = self.loader(path)
        if self._check_param_exists("transform"):
            sample = self.config_dict["transform"](sample)
        if self._check_param_exists("target_transform"):
            target = self.config_dict["target_transform"](target)

        return sample, target

    def __iter__(self):
        return self.data_loader.__iter__()

    def sampler(self, n):
        return [self.data_loader.__iter__() for i in range(n)]