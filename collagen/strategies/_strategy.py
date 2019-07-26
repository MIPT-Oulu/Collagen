from typing import Tuple

import torch
import torch.nn as nn
try:
    from torch.optim import Optimizer
except ImportError:
    from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from collagen.callbacks.visualization import ProgressbarVisualizer
from collagen.core import Callback
from collagen.core import Trainer, Session, Module
from collagen.core.utils import wrap_tuple
from collagen.data import DataProvider
from collagen.callbacks.metrics import RunningAverageMeter


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
        n_training_batches: int
            The number of training batches of each epoch. If None, the number of batches will be auto computed
        """

    def __init__(self, data_provider: DataProvider,
                 train_loader_names: Tuple[str] or str,
                 val_loader_names: Tuple[str] or str,
                 data_sampling_config: dict,
                 loss: nn.Module,
                 model: Module,
                 optimizer: Optimizer,
                 n_epochs: int or None = 100,
                 train_num_samples: Tuple[int] or int or None = None,
                 val_num_samples: Tuple[int] or int or None = None,
                 train_callbacks: Tuple[Callback] or Callback = None,
                 val_callbacks: Tuple[Callback] or Callback = None,
                 n_training_batches: int or None = None,
                 device: str or None = "cuda"):
        self.__data_provider: DataProvider = data_provider
        self.__loss: nn.Module = loss
        self.__optimizer: Optimizer = optimizer
        self.__model: Module = model

        self.__train_num_samples: Tuple[int] or int = train_num_samples
        self.__val_num_samples: Tuple[int] or int = val_num_samples

        self.__n_epochs: int = n_epochs

        self.__data_sampling_config = data_sampling_config
        self.__train_callbacks: Tuple[Callback] = wrap_tuple(train_callbacks)
        self.__val_callbacks: Tuple[Callback] = wrap_tuple(val_callbacks)
        self.__train_loader_names: Tuple[str] = wrap_tuple(train_loader_names)
        self.__val_loader_names: Tuple[str] = wrap_tuple(val_loader_names)
        self.__val_callbacks: Tuple[Callback] = wrap_tuple(val_callbacks)
        self.__train_callbacks: Tuple[Callback] = wrap_tuple(train_callbacks)

        if train_num_samples is None:
            self.__train_num_samples: Tuple[int] = tuple([1] * len(self.__train_loader_names))
        else:
            self.__train_num_samples: Tuple[int] = wrap_tuple(train_num_samples)

        if val_num_samples is None:
            self.__val_num_samples: Tuple[int] = tuple([1] * len(self.__val_loader_names))
        else:
            self.__val_num_samples: Tuple[int] = wrap_tuple(val_num_samples)

        if len(self.__train_loader_names) != len(self.__train_num_samples) or \
                len(self.__val_loader_names) != len(self.__val_num_samples):
            raise ValueError("The number of loaders and the number of sample quantities must be matched. "
                             "Train ({} vs {}), validation ({} vs {})".format(len(self.__train_loader_names),
                                                                              len(self.__train_num_samples),
                                                                              len(self.__val_loader_names),
                                                                              len(self.__val_num_samples)))

        self.__stage_names = ("train", "eval")
        self.__num_samples_by_stage = dict()
        self.__data_key_by_stage = dict()
        self.__target_key_by_stage = dict()
        self.__num_batches_by_stage = dict()
        for stage in self.__stage_names:
            self.__num_batches_by_stage[stage] = -1
            self.__data_key_by_stage[stage] = dict()
            self.__num_samples_by_stage[stage] = dict()
            self.__target_key_by_stage[stage] = dict()
            n_samples_dict = dict()

            data_keys = []
            target_keys = []
            data_loader_names = self.__data_sampling_config[stage]["data_provider"]
            for loader_name in data_loader_names:
                n_samples_dict[loader_name] = data_loader_names[loader_name]["num_samples"]
                n_batches = len(self.__data_provider.get_loader_by_name(loader_name))
                data_keys.append(data_loader_names[loader_name]["data_key"])
                target_keys.append(data_loader_names[loader_name]["target_key"])
                if self.__num_batches_by_stage[stage] < n_batches:
                    self.__num_batches_by_stage[stage] = n_batches

            self.__data_key_by_stage[stage] = tuple(data_keys)
            self.__target_key_by_stage[stage] = tuple(target_keys)

            self.__num_samples_by_stage[stage] = n_samples_dict

        if n_training_batches is not None:
            self.__num_batches_by_stage['train'] = n_training_batches

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")

        self.__model.to(self.__device)
        self.__loss.to(self.__device)

        self.__default_callbacks_train = (RunningAverageMeter(prefix='train', name='loss'),
                                          ProgressbarVisualizer(update_freq=1))
        self.__default_callbacks_eval = (RunningAverageMeter(prefix='eval', name='loss'),
                                         ProgressbarVisualizer(update_freq=1),)

        self.__train_callbacks = self._auto_add_default_callbacks(self.__default_callbacks_train,
                                                                  self.__train_callbacks)
        self.__val_callbacks = self._auto_add_default_callbacks(self.__default_callbacks_eval, self.__val_callbacks)

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

    def _auto_add_default_callbacks(self, d_cbs, cbs):
        added_train_cbs = []
        for d_cb in d_cbs:
            exist = False
            for cb in cbs:
                if cb.ctype == d_cb.ctype and cb.name == d_cb.name:
                    exist = True
                    break
            if not exist:
                added_train_cbs.append(d_cb)
        return cbs + tuple(added_train_cbs)

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

                self._call_callbacks_by_name('on_epoch_begin', epoch=epoch, stage=stage,
                                             n_epochs=self.__num_batches_by_stage[stage], trainer=self.__trainer)
                progress_bar = tqdm(range(self.__num_batches_by_stage[stage]),
                                    total=self.__num_batches_by_stage[stage],
                                    desc=f'Epoch [{epoch}] | {stage}::')
                for batch_i in progress_bar:
                    self._call_callbacks_by_name('on_sample_begin', epoch=epoch, stage=stage, batch_i=batch_i,
                                                 progress_bar=progress_bar, trainer=self.__trainer)
                    self.__data_provider.sample(**self.__num_samples_by_stage[stage])
                    self._call_callbacks_by_name('on_sample_end', epoch=epoch, stage=stage, batch_i=batch_i,
                                                 progress_bar=progress_bar, trainer=self.__trainer)
                    self._call_callbacks_by_name('on_batch_begin',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 trainer=self.__trainer)

                    getattr(self.__trainer, stage)(data_key=self.__data_key_by_stage[stage],
                                                   target_key=self.__target_key_by_stage[stage])

                    self._call_callbacks_by_name('on_batch_end',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 trainer=self.__trainer)
                self._call_callbacks_by_name('on_epoch_end', epoch=epoch, stage=stage,
                                             n_epochs=self.__num_batches_by_stage[stage], trainer=self.__trainer)
