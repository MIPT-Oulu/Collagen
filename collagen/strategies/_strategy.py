from collagen.core import Callback, Trainer, Session, Module
from collagen.data import DataProvider, ItemLoader, Splitter
import torch.nn as nn
from torch.optim import Optimizer
import pandas as pd
from typing import Tuple
import torch
from tqdm import tqdm


class Strategy(object):
    """
        Implements a part of the training loop by passing the available batches through the model.

        Parameters
        ----------
        data_provider : DataProvider
            Data provider. Controlled outside and samples mini-batches.
        train_loader_names : str or Tuple[str] or None
            Name of the training loader, which is a part of DataProvider.
        val_loader_names : str or Tuple[str] or None
            Name of the val loader, which is a part of DataProvider.
        session : Session
            Session to operate with
        train_callbacks : Tuple[Callback] or Callback or None
            Includes both metrics and callbacks. The callbacks can be schedulers,
            which allow to adjust the session parameters during training (can be useful for implementing super-convergence
            and stochastic weight averaging). On the other had the callbacks can also be meters batch-wise, which track
            losses / metrics during training.
        val_callbacks : Tuple[Callback] or Callback
            Includes both metrics and callbacks. Validation callbacks can be checkpointers, loggers,
            learning rate schedulers (E.g. reduce on plateau-like things). On the other hand,
             the callbacks can also be meters batch-wise, which compute metrics.
        """

    def __init__(self, data_provider: DataProvider,
                 train_loader_names: Tuple[str] or str,
                 val_loader_names: Tuple[str] or str,
                 data_key: str, target_key: str,
                 loss: nn.Module,
                 model: Module,
                 optimizer: Optimizer,
                 n_epochs: int or None = 100,
                 train_num_samples: Tuple[int] or int or None = None,
                 val_num_samples: Tuple[int] or int or None = None,
                 train_callbacks: Tuple[Callback] or Callback = None,
                 val_callbacks: Tuple[Callback] or Callback = None,
                 device: str or None = "cuda"):
        self.__data_provider: DataProvider = data_provider
        self.__loss: nn.Module = loss
        self.__optimizer: Optimizer = optimizer
        self.__model: Module = model

        self.__train_loader_names: Tuple[str] or str = train_loader_names
        self.__val_loader_names: Tuple[str] or str = val_loader_names

        self.__train_num_samples: Tuple[int] or int = train_num_samples
        self.__val_num_samples: Tuple[int] or int = val_num_samples

        self.__n_epochs: int = n_epochs

        self.__data_key = data_key
        self.__target_key = target_key

        self.__train_callbacks: Tuple[Callback] or Callback = train_callbacks
        self.__val_callbacks: Tuple[Callback] or Callback = val_callbacks

        if not isinstance(train_loader_names, Tuple):
            self.__train_loader_names: Tuple[str] = (train_loader_names, )

        if not isinstance(val_loader_names, Tuple):
            self.__val_loader_names: Tuple[str] = (val_loader_names, )

        if not isinstance(val_callbacks, Tuple):
            self.__val_callbacks: Tuple[Callback] or Callback = (val_callbacks, )

        if not isinstance(train_callbacks, Tuple):
            self.__train_callbacks: Tuple[Callback] = (train_callbacks, )

        if train_num_samples is None:
            self.__train_num_samples: Tuple[int] = tuple([1]*len(self.__train_loader_names))
        else:
            if not isinstance(train_num_samples, Tuple):
                self.__train_num_samples: Tuple[int] = (train_num_samples, )

        if val_num_samples is None:
            self.__val_num_samples: Tuple[int] = tuple([1]*len(self.__val_loader_names))
        else:
            if not isinstance(val_num_samples, Tuple):
                self.__val_num_samples: Tuple[int] = (val_num_samples, )

        if len(self.__train_loader_names) != len(self.__train_num_samples) or \
                len(self.__val_loader_names) != len(self.__val_num_samples):
            raise ValueError("The number of loaders and the number of sample quantities must be matched. "
                             "Train ({} vs {}), validation ({} vs {})".format(len(self.__train_loader_names),
                                                                       len(self.__train_num_samples),
                                                                       len(self.__val_loader_names),
                                                                       len(self.__val_num_samples)))

        self.__sampling_kwargs = {"train": dict(), "eval": dict()}
        self.__num_batches_by_stage = {"train": 0, "eval": 0}
        for stage in ['train', 'eval']:
            for i, num_samples in enumerate(self.__train_num_samples):
                if stage == "train":
                    self.__sampling_kwargs[stage][self.__train_loader_names[i]] = num_samples
                    self.__num_batches_by_stage[stage] = max(len(self.__data_provider.get_loader_by_name(self.__train_loader_names[i])), self.__num_batches_by_stage[stage])
                elif stage == "eval":
                    self.__sampling_kwargs[stage][self.__val_loader_names[i]] = num_samples
                    self.__num_batches_by_stage[stage] = max(len(self.__data_provider.get_loader_by_name(self.__val_loader_names[i])), self.__num_batches_by_stage[stage])
                else:
                    raise ValueError("Stage can only be `train` either `eval`, but found {}".format(stage))

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")

        self.__model.to(self.__device)
        self.__loss.to(self.__device)

    def run(self):
        se = Session(module=self.__model,
                     optimizer=self.__optimizer,
                     loss=self.__loss)

        trainer = Trainer(data_provider=self.__data_provider,
                          train_loader_names=self.__train_loader_names,
                          val_loader_names=self.__val_loader_names,
                          session=se,
                          train_callbacks=self.__train_callbacks,
                          val_callbacks=self.__val_callbacks)

        for epoch in range(self.__n_epochs):
            for cb in self.__train_callbacks:
                cb.on_epoch_begin(epoch=epoch,
                                  stage="train",
                                  data_provider=self.__data_provider,
                                  data_key=self.__data_key,
                                  target_key=self.__target_key,
                                  session=se)

            for cb in self.__val_callbacks:
                cb.on_epoch_begin(epoch=epoch,
                                  stage="eval",
                                  data_provider=self.__data_provider,
                                  data_key=self.__data_key,
                                  target_key=self.__target_key,
                                  session=se)
            for stage in ['train', 'eval']:
                for batch_i in tqdm(range(self.__num_batches_by_stage[stage]), total=self.__num_batches_by_stage[stage], desc=f'Epoch [{epoch}] | {stage}::'):
                    for cb in self.__train_callbacks:
                        cb.on_sample_begin(epoch=epoch,
                                           stage=stage,
                                           batch_index=batch_i,
                                           data_provider=self.__data_provider,
                                           data_key=self.__data_key,
                                           target_key=self.__target_key,
                                           session=se)
                    self.__data_provider.sample(**self.__sampling_kwargs[stage])
                    for cb in self.__train_callbacks:
                        cb.on_sample_end(epoch=epoch,
                                         stage=stage,
                                         batch_index=batch_i,
                                         data_provider=self.__data_provider,
                                         data_key=self.__data_key,
                                         target_key=self.__target_key,
                                         session=se)
                    getattr(trainer, stage)(data_key=self.__data_key, target_key=self.__target_key)

            for cb in self.__train_callbacks:
                cb.on_epoch_end(stage="train",
                                data_provider=self.__data_provider,
                                data_key=self.__data_key,
                                target_key=self.__target_key,
                                session=se)

            for cb in self.__val_callbacks:
                cb.on_epoch_end(stage="eval",
                                data_provider=self.__data_provider,
                                data_key=self.__data_key,
                                target_key=self.__target_key,
                                session=se)
