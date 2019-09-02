from typing import Tuple

import torch
import torch.nn as nn
try:
    from torch.optim import Optimizer
except ImportError:
    from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from collagen.core import Callback
from collagen.core import Trainer, Session, Module
from collagen.core.utils import wrap_tuple

from collagen.data import DataProvider

from collagen.callbacks.meters import RunningAverageMeter
from collagen.callbacks.logging.loggers import ProgressbarLogger


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
                 device: torch.device = torch.device('cpu')):
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

        self.__use_cuda = torch.cuda.is_available() and "cuda" in device.type
        self.__device = device

        self.__model.to(self.__device)
        self.__loss.to(self.__device)

        self.__default_callbacks_train = (RunningAverageMeter(prefix='train', name='loss'),
                                          ProgressbarLogger(update_freq=1, name='pbar/train'))
        self.__default_callbacks_eval = (RunningAverageMeter(prefix='eval', name='loss'),
                                         ProgressbarLogger(update_freq=1, name='pbar/eval'),)

        self.__train_callbacks = self._auto_add_default_callbacks(self.__default_callbacks_train,
                                                                  self.__train_callbacks)
        self.__val_callbacks = self._auto_add_default_callbacks(self.__default_callbacks_eval, self.__val_callbacks)

        self.__trainer = Trainer(data_provider=self.__data_provider,
                                 train_loader_names=self.__train_loader_names,
                                 val_loader_names=self.__val_loader_names,
                                 module=self.__model,
                                 optimizer=self.__optimizer,
                                 loss=self.__loss,
                                 train_callbacks=self.__train_callbacks,
                                 val_callbacks=self.__val_callbacks)

    @staticmethod
    def _auto_add_default_callbacks(d_cbs, cbs):
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


class MultiModelStrategy(object):
    """MultiModelStrategy implements the functionality to deal with multiple trainers and models. A helper yml file
    is deemed necessary for MultipleModelStrategy to work properly.
    """
    def __init__(self, data_provider: DataProvider,
                 data_sampling_config: dict,
                 strategy_config: dict,
                 device: torch.device,
                 callbacks: Tuple[Callback] or Callback = None,
                 n_epochs: int or None = 10,
                 n_train_batches: int or None = None,
                 trainers: Tuple[Trainer] or Trainer = None):
        """Constructor of MultiModelStrategy
        Parameters
        ----------
        data_provider: DataProvider
            Abstracts the access to all data for all trainers.
        data_sampling_config: dict
            Holds the configuration about how the data would be sampled
        strategy_config: dict
            Holds training configuration for multi-model-trainer. This is the core of Multi-model Strategy
        callbacks: Tuple[Callback]
            callback tuples to be executed inside the strategy main loop
        n_epochs: int
            Number of epochs to be trained.
        n_train_batches: int
            Number of training batches. Can be figured out from batch size and total data size
        trainers: Tuple[Trainer]
            Tuple of Trainer object. MultiModelStrategy will iterate through this tuple and execute them with proper
            callbacks and strategy configuration.
        Raises
        -------
        ValueError:
            The constructor will raise ValueError if data_provider or data_sampling_config or strategy_config or device
            is None

        """
        if data_provider is None or data_sampling_config is None or strategy_config is None or device is None:
            raise ValueError('data_provider or data_sampling_config or strategy_config or device cannot be None')
        # self.__stage names contains the learning stages e.g. 'train' and 'eval'. If you want to do only training but
        # no validation, go to the strategy.yml, remove 'eval' from 'stage_names'
        self.__stage_names = strategy_config['stage_names']
        # self.__model_trainer_names contains the name of trainable models. A trainable model may contain multiple
        # NN models
        self.__model_trainer_names = strategy_config['model_trainer_names']
        self.__n_epochs = n_epochs
        self.__data_provider = data_provider
        self.__callbacks = wrap_tuple(callbacks)
        # self.__accumulate_grad is retrieved from strategy config yml file. It contains boolean value for each trainer
        # in the trainers tuple. This is a mandatory field in the yml file
        self.__accumulate_grad = strategy_config['accumulate_grad']
        self.__data_sampling_config = data_sampling_config

        # self.__num_samples_by_stage is a dictionary holding the number of batches per stage
        self.__num_samples_by_stage = dict()
        # self.__data_key_by_stage is a dictionary of dictionary. Key to the first dictionary is the stage name and
        # key to the embedded dictionary is the trainable model name, the final value is data key to be used retrieving
        # data from the data_provider
        self.__data_key_by_stage = dict()
        # self.__target_key_by_stage is similar to __data_key_by_stage, as the name suggests the final value is the
        # target key to be used to retrieve target value from data_provider
        self.__target_key_by_stage = dict()
        self.__num_batches_by_stage = dict()

        for stage in self.__stage_names:
            self.__num_batches_by_stage[stage] = -1
            self.__data_key_by_stage[stage] = dict()
            self.__num_samples_by_stage[stage] = dict()
            self.__target_key_by_stage[stage] = dict()
            # iterate through each trainable model trainer
            for model_trainer_name in self.__model_trainer_names:
                # n_sample_dict is a local variable of dict type. trainable model name and number of samples for this
                # trainer are saved as key: value pair.
                # readers be aware that model_trainer_name is the name of the trainer which might contain multiple
                # NN models
                n_samples_dict = dict()
                data_keys = []
                target_keys = []
                if model_trainer_name in self.__data_sampling_config[stage]['data_provider']:
                    data_loader_names = self.__data_sampling_config[stage]['data_provider'][model_trainer_name]

                else:
                    continue

                for loader_name in data_loader_names:
                    n_samples_dict[loader_name] = data_loader_names[loader_name]['num_samples']
                    n_batches = len(self.__data_provider.get_loader_by_name(loader_name))
                    data_keys.append(data_loader_names[loader_name]['data_key'])
                    target_keys.append(data_loader_names[loader_name]['target_key'])

                    if self.__num_batches_by_stage[stage] < n_batches:
                        self.__num_batches_by_stage[stage] = n_batches

                self.__data_key_by_stage[stage][model_trainer_name] = tuple(data_keys)
                self.__target_key_by_stage[stage][model_trainer_name] = tuple(target_keys)

                self.__num_samples_by_stage[stage][model_trainer_name] = n_samples_dict

        if n_train_batches is not None and n_train_batches > 0:
            self.__num_batches_by_stage['train'] = n_train_batches

        self.__device = device
        self.__trainers = dict()

        # save the trainers into dictionary according to model trainer name for easy access
        for name, trainer in zip(self.__model_trainer_names, trainers):
            self.__trainers[name] = trainer

        self.__default_callbacks_train = dict()
        self.__default_callbacks_eval = dict()
        strategy_callback_tuple = ()
        for name, trainer in zip(self.__model_trainer_names, trainers):
            self.__trainers[name] = trainer
            self.__default_callbacks_train[name] = RunningAverageMeter(prefix=f'train/{name}', name='loss')
            self.__default_callbacks_eval[name] = RunningAverageMeter(prefix=f'eval/{name}', name='loss')
            self.__trainers[name].add_train_callbacks(self.__default_callbacks_train[name])
            self.__trainers[name].add_eval_callbacks(self.__default_callbacks_eval[name])
            strategy_callback_tuple += wrap_tuple(trainer.model)

        self.__default_strategy_callback = (ProgressbarLogger(update_freq=1), )

        self.__callbacks += self.__default_strategy_callback

    def _call_callbacks_by_name(self, cb_func_name, **kwargs):
        """
        _call_callbacks_by_name is a private function to be called inside the class. This function traverses all the
        trainers to check if they have the callback function to be called. If the callback is found it is called.
        Afterwards, it searches the self.__callback, a private variable holding manually provided callbacks, if the
        provided callback is found here, it is also called.

        Parameters
        ----------
        cb_func_name: str
            name of the call_back_function to be called
        kwargs: list or tuple
            argument for the callback function
        """
        for model_name in self.__model_trainer_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(kwargs['stage']):
                if hasattr(cb, cb_func_name):
                    getattr(cb, cb_func_name)(strategy=self, **kwargs)
        for cb in self.__callbacks:
            if hasattr(cb, cb_func_name):
                getattr(cb, cb_func_name)(strategy=self, **kwargs)

    def get_callbacks_by_name(self, name, stage):
        """
        get_callbacks_by_trainer_name is a public function which only retrieves some callback but does not call it
        Parameter
        ---------
        name: str
            name of the trainers where the callback function would be searched
        stage: str
            name of the learning stage where the call back function name would be searched.
        """
        if name in self.__model_trainer_names:
            return self.__trainers[name].get_callbacks_by_stage(stage)
        else:
            if name == 'minibatch':
                cbs = ()
                for name in self.__model_trainer_names:
                    cbs += self.__trainers[name].get_callbacks_by_stage(stage)
                return cbs
            elif name == 'batch':
                return self.__callbacks
            else:  # return all
                cbs = ()
                for name in self.__model_trainer_names:
                    cbs += self.__trainers[name].get_callbacks_by_stage(stage)
                return cbs + self.__callbacks

    # the great and famous run function
    def run(self):
        """ run function runs the strategy, the epoch iteration is done here. Inside each epoch, for each trainable
         model data is sampled and each trainable model is run with the sampled data.
        """
        for epoch in range(self.__n_epochs):
            for stage in self.__stage_names:
                self._call_callbacks_by_name(cb_func_name='on_epoch_begin', epoch=epoch, stage=stage,
                                             n_epochs=self.__n_epochs)
                progress_bar = tqdm(range(self.__num_batches_by_stage[stage]), total=self.__num_batches_by_stage[stage],
                                    desc=f'Epoch [{epoch}][{stage}]::')
                for batch_i in progress_bar:
                    self._call_callbacks_by_name(cb_func_name='on_batch_begin',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 n_batches=self.__num_batches_by_stage[stage])

                    # for scalability data sampling needs to be done every iteration
                    for model_name in self.__model_trainer_names:
                        self._call_callbacks_by_name(cb_func_name='on_sample_begin',
                                                     progress_bar=progress_bar,
                                                     epoch=epoch,
                                                     n_epochs=self.__n_epochs,
                                                     stage=stage,
                                                     batch_i=batch_i,
                                                     n_batches=self.__num_batches_by_stage[stage])
                        # sample data for current trainable model
                        self.__data_provider.sample(**self.__num_samples_by_stage[stage][model_name])

                        self._call_callbacks_by_name(cb_func_name='on_sample_end',
                                                     progress_bar=progress_bar,
                                                     epoch=epoch,
                                                     n_epochs=self.__n_epochs,
                                                     stage=stage,
                                                     batch_i=batch_i,
                                                     n_batches=self.__num_batches_by_stage[stage])
                        # get callable function from current model depending on the stage
                        getattr(self.__trainers[model_name], stage)(data_key=
                                                                    self.__data_key_by_stage[stage][model_name],
                                                                    target_key=
                                                                    self.__target_key_by_stage[stage][model_name],
                                                                    accumulate_grad=self.__accumulate_grad[model_name])

                    self._call_callbacks_by_name(cb_func_name='on_batch_end',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 n_batches=self.__num_batches_by_stage[stage])

                self._call_callbacks_by_name(cb_func_name='on_epoch_end', epoch=epoch, n_epochs=self.__n_epochs,
                                             stage=stage, n_batches=self.__num_batches_by_stage[stage])