from typing import Tuple, Dict
import torch


try:
    from torch.optim import Optimizer
except ImportError:
    from torch.optim.optimizer import Optimizer

from tqdm import tqdm

from collagen.core import Callback
from collagen.core import Session
from collagen.core.utils import wrap_tuple
from collagen.data import DataProvider
from collagen.callbacks.logging.loggers import ProgressbarLogger


class Strategy(object):
    """Strategy implements the functionality to deal with multiple sessions and models. A helper yml file
    is deemed necessary for MultipleModelStrategy to work properly.
    """

    def __init__(self, data_provider: DataProvider,
                 data_sampling_config: dict,
                 device: torch.device,
                 strategy_config: dict = None,
                 callbacks: Tuple[Callback] or Callback or None = None,
                 n_epochs: int or None = 10,
                 n_train_batches: int or None = None,
                 train_batchs_choice: str = 'max',
                 sessions: Dict[str, Session] or Session = None,
                 distributed=False, use_apex=False):
        """Constructor of Strategy
        Parameters
        ----------
        data_provider: DataProvider
            Abstracts the access to all data for all sessions.
        data_sampling_config: dict
            Holds the configuration about how the data would be sampled
        strategy_config: dict
            Holds training configuration for multi-model-stepper. This is the core of Multi-model Strategy
        callbacks: Tuple[Callback]
            callback tuples to be executed inside the strategy main loop
        n_epochs: int
            Number of epochs to be trained.
        n_train_batches: int
            Number of training batches. Can be figured out from batch size and total data size
        sessions: Dict[str, Session]
            Dict of Session objects. Strategy will iterate through this dict by name and execute them with proper
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
        self.__stage_names = wrap_tuple(strategy_config['stage_names'])
        # self.__train_model_names contains the name of trainable models. A trainable model may contain multiple
        # NN models

        # TODO: Is it necessary to be a set instead of tuple?
        self.__model_names_by_stage = dict()
        self.__model_names_by_stage['train'] = wrap_tuple(data_sampling_config.train.data_provider.keys())
        self.__model_names_by_stage['eval'] = wrap_tuple(data_sampling_config.eval.data_provider.keys())
        self.__train_starts_at_epoch = strategy_config['train_starts_at_epoch']
        self.__n_epochs = n_epochs
        self.__data_provider = data_provider
        self.__callbacks = wrap_tuple(callbacks)
        # self.__accumulate_grad is retrieved from strategy config yml file. It contains boolean value for each stepper
        # in the sessionss tuple. This is a mandatory field in the yml file
        self.__accumulate_grad = strategy_config['accumulate_grad']
        self.__accumulate_grad_in_iter = strategy_config['accumulate_grad_in_iter']
        self.__data_sampling_config = data_sampling_config

        sessions = wrap_tuple(sessions)

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
        self.__distributed = distributed
        self.__use_apex = use_apex
        for stage in self.__stage_names:
            self.__num_batches_by_stage[stage] = None
            self.__data_key_by_stage[stage] = dict()
            self.__num_samples_by_stage[stage] = dict()
            self.__target_key_by_stage[stage] = dict()

            model_names = self.__model_names_by_stage[stage]

            # iterate through each trainable model stepper
            for model_name in model_names:
                # n_sample_dict is a local variable of dict type. trainable model name and number of samples for this
                # stepper are saved as key: value pair.
                # readers be aware that model_name is the name of the stepper which might contain multiple
                # NN models
                n_samples_dict = dict()
                data_keys = []
                target_keys = []
                if model_name in self.__data_sampling_config[stage]['data_provider']:
                    data_loader_names = self.__data_sampling_config[stage]['data_provider'][model_name]

                else:
                    continue

                for loader_name in data_loader_names:
                    n_samples_dict[loader_name] = data_loader_names[loader_name]['batches_per_iter']
                    n_batches = len(self.__data_provider.get_loader_by_name(loader_name))
                    data_keys.append(data_loader_names[loader_name]['data_key'])
                    target_keys.append(data_loader_names[loader_name]['target_key'])

                    if self.__num_batches_by_stage[stage] is None:
                        self.__num_batches_by_stage[stage] = n_batches
                    elif (train_batchs_choice == 'max' and self.__num_batches_by_stage[stage] < n_batches):
                        self.__num_batches_by_stage[stage] = n_batches
                    elif train_batchs_choice == 'min' and self.__num_batches_by_stage[stage] > n_batches:
                        self.__num_batches_by_stage[stage] = n_batches

                self.__data_key_by_stage[stage][model_name] = tuple(data_keys)
                self.__target_key_by_stage[stage][model_name] = tuple(target_keys)

                self.__num_samples_by_stage[stage][model_name] = n_samples_dict

        if n_train_batches is not None and n_train_batches > 0:
            self.__num_batches_by_stage['train'] = n_train_batches

        self.__device = device
        self.__sessions = dict()

        # save the sessions into dictionary according to model stepper name for easy access
        optimizers = dict()
        for name in self.__model_names_by_stage['train']:
            self.__sessions[name] = sessions[name]
            if self.__sessions[name].data_provider is None:
                self.__sessions[name].data_provider = self.__data_provider

            if len(sessions) == 1:
                optimizers = self.__sessions[name].optimizer
            else:
                optimizers[name] = self.__sessions[name].optimizer

        if self.__use_apex:
            from collagen.parallel._apex import first_gpu_or_cpu_in_use
            if first_gpu_or_cpu_in_use(self.__device):
                self.__default_strategy_callback = (ProgressbarLogger(update_freq=1, optimizers=optimizers),)
            else:
                self.__default_strategy_callback = ()
        else:
            self.__default_strategy_callback = (ProgressbarLogger(update_freq=1, optimizers=optimizers),)

        self.__callbacks += self.__default_strategy_callback

    @property
    def data_provider(self):
        return self.__data_provider

    def _call_callbacks_by_name(self, cb_func_name, **kwargs):
        """
        _call_callbacks_by_name is a private function to be called inside the class. This function traverses all the
        sessions to check if they have the callback function to be called. If the callback is found it is called.
        Afterwards, it searches the self.__callback, a private variable holding manually provided callbacks, if the
        provided callback is found here, it is also called.

        Parameters
        ----------
        cb_func_name: str
            name of the call_back_function to be called
        kwargs: list or tuple
            argument for the callback function
        """

        for model_name in self.__model_names_by_stage[kwargs['stage']]:
            for cb in getattr(self.__sessions[model_name], f'get_callbacks_by_stage')(kwargs['stage']):
                if hasattr(cb, cb_func_name):
                    getattr(cb, cb_func_name)(strategy=self, **kwargs)
        for cb in self.__callbacks:
            if hasattr(cb, cb_func_name):
                getattr(cb, cb_func_name)(strategy=self, **kwargs)

    def get_callbacks_by_name(self, name, stage):
        """
        get_callbacks_by_name is a public function which only retrieves some callback but does not call it
        Parameter
        ---------
        name: str
            name of the sessions where the callback function would be searched
        stage: str
            name of the learning stage where the call back function name would be searched.
        """
        if name in self.__model_names_by_stage[stage]:
            return self.__sessions[name].get_callbacks_by_stage(stage)
        else:
            if name == 'minibatch':
                cbs = ()
                for name in self.__model_names_by_stage[stage]:
                    cbs += self.__sessions[name].get_callbacks_by_stage(stage)
                return cbs
            elif name == 'batch':
                return self.__callbacks
            else:  # return all
                cbs = ()
                for name in self.__model_names_by_stage[stage]:
                    cbs += self.__sessions[name].get_callbacks_by_stage(stage)
                return cbs + self.__callbacks

    # the great and famous run function
    def run(self):
        """ run function runs the strategy, the epoch iteration is done here. Inside each epoch, for each trainable
         model data is sampled and each trainable model is run with the sampled data.
        """
        for epoch in range(self.__n_epochs):
            self.__data_provider.set_epoch(epoch)
            for stage in self.__stage_names:

                model_names = self.__model_names_by_stage[stage]
                self._call_callbacks_by_name(cb_func_name='on_epoch_begin', epoch=epoch, stage=stage,
                                             n_epochs=self.__n_epochs)
                if not self.__use_apex or (self.__use_apex and first_gpu_or_cpu_in_use(self.__device)):
                    model_names = [_name for _name in model_names if self.__train_starts_at_epoch[_name] <= epoch]
                    model_names_str = "-".join(model_names)
                    progress_bar = tqdm(range(self.__num_batches_by_stage[stage]), total=self.__num_batches_by_stage[stage],
                                        desc=f'Epoch [{epoch}][{stage}]::[{model_names_str}]')
                else:
                    progress_bar = range(self.__num_batches_by_stage[stage])
                for batch_i in progress_bar:
                    self._call_callbacks_by_name(cb_func_name='on_batch_begin',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 n_batches=self.__num_batches_by_stage[stage])

                    # for scalability data sampling needs to be done every iteration
                    for model_name in model_names:
                        if self.__train_starts_at_epoch[model_name] > epoch and stage == "train":
                            continue

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
                        getattr(self.__sessions[model_name], stage)(data_key=
                                                                    self.__data_key_by_stage[stage][model_name],
                                                                    target_key=
                                                                    self.__target_key_by_stage[stage][model_name],
                                                                    accumulate_grad=self.__accumulate_grad[model_name],
                                                                    accumulate_grad_in_iter=self.__accumulate_grad_in_iter[model_name])

                    self._call_callbacks_by_name(cb_func_name='on_batch_end',
                                                 progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i,
                                                 n_batches=self.__num_batches_by_stage[stage])

                self._call_callbacks_by_name(cb_func_name='on_epoch_end', epoch=epoch, n_epochs=self.__n_epochs,
                                             stage=stage, n_batches=self.__num_batches_by_stage[stage])
