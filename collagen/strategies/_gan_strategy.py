from collagen.core import Trainer, Callback
from collagen.core.utils import to_tuple
from collagen.data import DataProvider
from collagen.metrics import RunningAverageMeter
from collagen.callbacks import OnDiscriminatorBatchFreezer, OnGeneratorBatchFreezer
from collagen.callbacks import ProgressbarVisualizer, OnSamplingFreezer
from typing import Tuple
import torch
from tqdm import tqdm


class GANStrategy(object):
    """
        Implements a part of the training loop by passing the available batches through the model.

        Parameters
        ----------
        data_provider : DataProvider
            Data provider. Controlled outside and samples mini-batches.
    """

    def __init__(self, data_provider: DataProvider,
                 data_sampling_config: dict,
                 d_trainer: Trainer, g_trainer: Trainer,
                 n_epochs: int or None = 100,
                 callbacks: Tuple[Callback] or Callback = None,
                 device: str or None = "cuda",
                 n_training_batches: int or None=None):
        """Implements a part of the training GAN loop by passing the available batches through the model.

        Parameters
        ----------
        data_provider: DataProvider
            Provides batches of data to D and G models
        data_sampling_config: dict
            Configuration of the itemloader names and the corresponding numbers of samples
        d_trainer: Trainer
            Trainer of Discriminative model
        g_trainer: Trainer
            Trainer of Generative model
        n_epochs: int
            The number of epochs
        callbacks: Callback or Tuple[Callback]
            Callbacks at strategy-level
        device: device
            Device on which forwarding and backwarding take place
        n_training_batches: int
            The number of training batches of each epoch. If None, the number of batches will be auto computed
        """
        self.__stage_names = ("train", "eval")
        self.__model_names = ("G", "D")
        self.__n_epochs = n_epochs
        self.__callbacks = to_tuple(callbacks)
        self.__data_provider = data_provider

        self.__data_sampling_config = data_sampling_config

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
            for model_name in self.__model_names:
                data_keys = []
                target_keys = []
                if model_name in self.__data_sampling_config[stage]["data_provider"]:
                    data_loader_names = self.__data_sampling_config[stage]["data_provider"][model_name]
                else:
                    continue
                for loader_name in data_loader_names:
                    n_samples_dict[loader_name] = data_loader_names[loader_name]["num_samples"]
                    n_batches = len(self.__data_provider.get_loader_by_name(loader_name))
                    data_keys.append(data_loader_names[loader_name]["data_key"])
                    target_keys.append(data_loader_names[loader_name]["target_key"])
                    if self.__num_batches_by_stage[stage] < n_batches:
                        self.__num_batches_by_stage[stage] = n_batches

                self.__data_key_by_stage[stage][model_name] = tuple(data_keys)
                self.__target_key_by_stage[stage][model_name] = tuple(target_keys)

            self.__num_samples_by_stage[stage] = n_samples_dict

        if n_training_batches is not None:
            self.__num_batches_by_stage['train'] = n_training_batches

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")
        self.__trainers = {"D": d_trainer, "G": g_trainer}

        # Default minibatch level callbacks
        self.__default_g_callbacks_train = (RunningAverageMeter(prefix="train/G", name="loss"),
                                            OnGeneratorBatchFreezer(modules=d_trainer.model))
        self.__default_d_callbacks_train = (RunningAverageMeter(prefix="train/D", name="loss"),
                                            OnDiscriminatorBatchFreezer(modules=g_trainer.model))
        self.__default_g_callbacks_eval = RunningAverageMeter(prefix="eval/G", name="loss")
        self.__default_d_callbacks_eval = RunningAverageMeter(prefix="eval/D", name="loss")

        self.__trainers["G"].add_train_callbacks(self.__default_g_callbacks_train)
        self.__trainers["D"].add_train_callbacks(self.__default_d_callbacks_train)
        self.__trainers["G"].add_eval_callbacks(self.__default_g_callbacks_eval)
        self.__trainers["D"].add_eval_callbacks(self.__default_d_callbacks_eval)

        # Default epoch level callbacks
        self.__default_st_callbacks = (OnSamplingFreezer(modules=to_tuple(d_trainer.model)+to_tuple(g_trainer.model)),
                                       ProgressbarVisualizer(update_freq=1))
        self.__callbacks += self.__default_st_callbacks


    def _call_callbacks_by_name(self, cb_func_name, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(kwargs['stage']):
                getattr(cb, cb_func_name)(strategy=self, **kwargs)

        for cb in self.__callbacks:
            getattr(cb, cb_func_name)(strategy=self, **kwargs)


    def get_callbacks_by_name(self, name, stage):
        if name == "D":
            return self.__trainers[name].get_callbacks_by_stage(stage)
        elif name == "G":
            return self.__trainers[name].get_callbacks_by_stage(stage)
        elif name == "minibatch":
            return self.__trainers["D"].get_callbacks_by_stage(stage) + self.__trainers["G"].get_callbacks_by_stage(stage)
        elif name == "batch":
            return self.__callbacks
        elif name == "all":
            return self.__trainers["D"].get_callbacks_by_stage(stage) \
                   + self.__trainers["G"].get_callbacks_by_stage(stage) \
                   + self.__callbacks

    def run(self):
        for epoch in range(self.__n_epochs):
            for stage in self.__stage_names:
                self._call_callbacks_by_name(cb_func_name='on_epoch_begin', epoch=epoch, stage=stage,
                                               n_epochs=self.__n_epochs)

                progress_bar = tqdm(range(self.__num_batches_by_stage[stage]), total=self.__num_batches_by_stage[stage],
                                    desc=f'Epoch [{epoch}][{stage}]::')
                for batch_i in progress_bar:
                    self._call_callbacks_by_name(cb_func_name='on_sample_begin',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self.__data_provider.sample(**self.__num_samples_by_stage[stage])
                    self._call_callbacks_by_name(cb_func_name='on_sample_end',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self._call_callbacks_by_name(cb_func_name='on_batch_begin',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self._call_callbacks_by_name(cb_func_name='on_gan_d_batch_begin',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    if "D" in self.__data_key_by_stage[stage]:
                        getattr(self.__trainers["D"], stage)(data_key=self.__data_key_by_stage[stage]["D"],
                                                             target_key=self.__target_key_by_stage[stage]["D"])
                    self._call_callbacks_by_name(cb_func_name='on_gan_d_batch_end',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)
                    self._call_callbacks_by_name(cb_func_name='on_gan_g_batch_begin',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    if "G" in self.__data_key_by_stage[stage]:
                        getattr(self.__trainers["G"], stage)(data_key=self.__data_key_by_stage[stage]["G"],
                                                             target_key=self.__target_key_by_stage[stage]["G"])
                    self._call_callbacks_by_name(cb_func_name='on_gan_d_batch_end',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self._call_callbacks_by_name(cb_func_name='on_batch_end',
                                                   progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)


                self._call_callbacks_by_name(cb_func_name='on_epoch_end', epoch=epoch, n_epochs=self.__n_epochs,
                                               stage=stage)
