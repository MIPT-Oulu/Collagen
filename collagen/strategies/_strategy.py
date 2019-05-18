import torch.nn as nn
from torch.optim import Optimizer
from typing import Tuple
import torch
from tqdm import tqdm
from collagen.core import Trainer, Session, Module
from collagen.core.utils import to_tuple
from collagen.core import Callback
from collagen.data import DataProvider, ItemLoader, Splitter


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

        self.__train_callbacks: Tuple[Callback] = to_tuple(train_callbacks)
        self.__val_callbacks: Tuple[Callback] = to_tuple(val_callbacks)
        self.__train_loader_names: Tuple[str] = to_tuple(train_loader_names)
        self.__val_loader_names: Tuple[str] = to_tuple(val_loader_names)
        self.__val_callbacks: Tuple[Callback] = to_tuple(val_callbacks)
        self.__train_callbacks: Tuple[Callback] = to_tuple(train_callbacks)

        if train_num_samples is None:
            self.__train_num_samples: Tuple[int] = tuple([1] * len(self.__train_loader_names))
        else:
            self.__train_num_samples: Tuple[int] = to_tuple(train_num_samples)

        if val_num_samples is None:
            self.__val_num_samples: Tuple[int] = tuple([1] * len(self.__val_loader_names))
        else:
            self.__val_num_samples: Tuple[int] = to_tuple(val_num_samples)

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
                    self.__num_batches_by_stage[stage] = max(
                        len(self.__data_provider.get_loader_by_name(self.__train_loader_names[i])),
                        self.__num_batches_by_stage[stage])
                elif stage == "eval":
                    self.__sampling_kwargs[stage][self.__val_loader_names[i]] = num_samples
                    self.__num_batches_by_stage[stage] = max(
                        len(self.__data_provider.get_loader_by_name(self.__val_loader_names[i])),
                        self.__num_batches_by_stage[stage])
                else:
                    raise ValueError("Stage can only be `train` either `eval`, but found {}".format(stage))

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")

        self.__model.to(self.__device)
        self.__loss.to(self.__device)

        self.__session = Session(module=self.__model,
                                 optimizer=self.__optimizer,
                                 loss=self.__loss)

        self.__trainer = Trainer(data_provider=self.__data_provider,
                                 train_loader_names=self.__train_loader_names,
                                 val_loader_names=self.__val_loader_names,
                                 module=self.__model,
                                 optimizer=self.__optimizer,
                                 loss=self.__loss,
                                 train_callbacks=self.__train_callbacks,
                                 val_callbacks=self.__val_callbacks)

    def get_callbacks_by_name(self, name, stage):
        if name == "minibatch" or name == "all":
            return self.get_callbacks_by_stage(stage)
        else:
            raise ValueError("Only support `minibatch` or `all`, but got {}".format(name))

    def get_callbacks_by_stage(self, stage):
        if stage == "train":
            return self.__train_callbacks
        elif stage == "eval":
            return self.__val_callbacks
        else:
            raise ValueError("Only support `train` and `eval` stage, but got {}".format(stage))

    def _call_callbacks_by_name(self, cb_func_name, **kwargs):
        for cb in self.get_callbacks_by_stage(kwargs['stage']):
            getattr(cb, cb_func_name)(strategy=self, **kwargs)

    def run(self):
        for epoch in range(self.__n_epochs):
            for stage in ['train', 'eval']:

                self._call_callbacks_by_name('on_epoch_begin', epoch=epoch, stage=stage)
                progress_bar = tqdm(range(self.__num_batches_by_stage[stage]),
                                    total=self.__num_batches_by_stage[stage],
                                    desc=f'Epoch [{epoch}] | {stage}::')
                for batch_i in progress_bar:
                    self._call_callbacks_by_name('on_sample_begin', epoch=epoch, stage=stage, batch_i=batch_i,
                                                 progress_bar=progress_bar)
                    self.__data_provider.sample(**self.__sampling_kwargs[stage])
                    self._call_callbacks_by_name('on_sample_end', epoch=epoch, stage=stage, batch_i=batch_i,
                                                 progress_bar=progress_bar)
                    self._call_callbacks_by_name('on_batch_begin',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i)

                    getattr(self.__trainer, stage)(data_key=self.__data_key, target_key=self.__target_key)

                    self._call_callbacks_by_name('on_batch_end',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i)
            self._call_callbacks_by_name('on_epoch_end', epoch=epoch, stage=stage)
