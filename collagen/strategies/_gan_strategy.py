from collagen.core import Trainer, Callback
from collagen.core.utils import to_tuple
from collagen.data import DataProvider

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
                 device: str or None = "cuda"):
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
            Device on which forward and backward are performed
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

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")
        self.__trainers = {"D": d_trainer, "G": g_trainer}

    def _on_batch_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_batch_begin(epoch=epoch,
                                  n_epochs=n_epochs,
                                  batch_i=batch_i,
                                  progress_bar=progress_bar,
                                  stage=stage,
                                  strategy=self)

        for cb in self.__callbacks:
            cb.on_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage=stage,
                              strategy=self)

    def _on_batch_end_callbacks(self, epoch, stage, n_epochs, batch_i, progress_bar):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_batch_end(epoch=epoch,
                                n_epochs=n_epochs,
                                batch_i=batch_i,
                                progress_bar=progress_bar,
                                stage="generate",
                                strategy=self)

        for cb in self.__callbacks:
            cb.on_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage=stage,
                            strategy=self)

    def _on_epoch_begin_callbacks(self, epoch, n_epochs, stage):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_epoch_begin(epoch=epoch,
                                  n_epochs=n_epochs,
                                  stage=stage,
                                  strategy=self)

        for cb in self.__callbacks:
            cb.on_epoch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              stage=stage,
                              strategy=self)

    def _on_epoch_end_callbacks(self, epoch, stage, n_epochs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_epoch_end(epoch=epoch,
                                n_epochs=n_epochs,
                                stage=stage,
                                strategy=self)

        for cb in self.__callbacks:
            cb.on_epoch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            stage=stage,
                            strategy=self)

    def _on_sample_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_sample_begin(epoch=epoch,
                                   n_epochs=n_epochs,
                                   batch_i=batch_i,
                                   progress_bar=progress_bar,
                                   stage=stage,
                                   batch_index=batch_i,
                                   strategy=self)

        for cb in self.__callbacks:
            cb.on_sample_begin(epoch=epoch,
                               n_epochs=n_epochs,
                               batch_i=batch_i,
                               progress_bar=progress_bar,
                               stage=stage,
                               batch_index=batch_i,
                               strategy=self)

    def _on_sample_end_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_sample_end(epoch=epoch,
                                 n_epochs=n_epochs,
                                 batch_i=batch_i,
                                 progress_bar=progress_bar,
                                 stage=stage,
                                 batch_index=batch_i,
                                 strategy=self)

        for cb in self.__callbacks:
            cb.on_sample_end(epoch=epoch,
                             n_epochs=n_epochs,
                             batch_i=batch_i,
                             progress_bar=progress_bar,
                             stage=stage,
                             batch_index=batch_i,
                             strategy=self)

    def _on_d_batch_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_gan_d_batch_begin(epoch=epoch,
                                        n_epochs=n_epochs,
                                        batch_i=batch_i,
                                        progress_bar=progress_bar,
                                        stage=stage,
                                        batch_index=batch_i,
                                        strategy=self)

        for cb in self.__callbacks:
            cb.on_gan_d_batch_begin(epoch=epoch,
                                    n_epochs=n_epochs,
                                    batch_i=batch_i,
                                    progress_bar=progress_bar,
                                    stage=stage,
                                    batch_index=batch_i,
                                    strategy=self)

    def _on_d_batch_end_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_gan_d_batch_end(epoch=epoch,
                                      n_epochs=n_epochs,
                                      batch_i=batch_i,
                                      progress_bar=progress_bar,
                                      stage=stage,
                                      batch_index=batch_i,
                                      strategy=self)

        for cb in self.__callbacks:
            cb.on_gan_d_batch_end(epoch=epoch,
                                  n_epochs=n_epochs,
                                  batch_i=batch_i,
                                  progress_bar=progress_bar,
                                  stage=stage,
                                  batch_index=batch_i,
                                  strategy=self)

    def _on_g_batch_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_gan_g_batch_begin(epoch=epoch,
                                        n_epochs=n_epochs,
                                        batch_i=batch_i,
                                        progress_bar=progress_bar,
                                        stage=stage,
                                        batch_index=batch_i,
                                        strategy=self)

        for cb in self.__callbacks:
            cb.on_gan_g_batch_begin(epoch=epoch,
                                    n_epochs=n_epochs,
                                    batch_i=batch_i,
                                    progress_bar=progress_bar,
                                    stage=stage,
                                    batch_index=batch_i,
                                    strategy=self)

    def _on_g_batch_end_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_callbacks_by_stage')(stage):
                cb.on_gan_g_batch_end(epoch=epoch,
                                      n_epochs=n_epochs,
                                      batch_i=batch_i,
                                      progress_bar=progress_bar,
                                      stage=stage,
                                      batch_index=batch_i,
                                      strategy=self)

        for cb in self.__callbacks:
            cb.on_gan_g_batch_end(epoch=epoch,
                                  n_epochs=n_epochs,
                                  batch_i=batch_i,
                                  progress_bar=progress_bar,
                                  stage=stage,
                                  batch_index=batch_i,
                                  strategy=self)

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
                self._on_epoch_begin_callbacks(epoch=epoch, stage=stage, n_epochs=self.__n_epochs)
                # DEBUG
                # self.__num_batches_by_stage[stage] = 5
                progress_bar = tqdm(range(self.__num_batches_by_stage[stage]), total=self.__num_batches_by_stage[stage],
                                    desc=f'Epoch [{epoch}][{stage}]::')
                for batch_i in progress_bar:
                    self._on_sample_begin_callbacks(progress_bar=progress_bar,
                                                    epoch=epoch,
                                                    n_epochs=self.__n_epochs,
                                                    stage=stage,
                                                    batch_i=batch_i)

                    self.__data_provider.sample(**self.__num_samples_by_stage[stage])

                    self._on_sample_end_callbacks(progress_bar=progress_bar,
                                                  epoch=epoch,
                                                  n_epochs=self.__n_epochs,
                                                  stage=stage,
                                                  batch_i=batch_i)

                    self._on_batch_begin_callbacks(progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self._on_d_batch_begin_callbacks(progress_bar=progress_bar,
                                                     epoch=epoch,
                                                     n_epochs=self.__n_epochs,
                                                     stage=stage,
                                                     batch_i=batch_i)
                    if "D" in self.__data_key_by_stage[stage]:
                        getattr(self.__trainers["D"], stage)(data_key=self.__data_key_by_stage[stage]["D"],
                                                             target_key=self.__target_key_by_stage[stage]["D"])

                    self._on_d_batch_end_callbacks(progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)
                    self._on_g_batch_begin_callbacks(progress_bar=progress_bar,
                                                     epoch=epoch,
                                                     n_epochs=self.__n_epochs,
                                                     stage=stage,
                                                     batch_i=batch_i)
                    if "G" in self.__data_key_by_stage[stage]:
                        getattr(self.__trainers["G"], stage)(data_key=self.__data_key_by_stage[stage]["G"],
                                                             target_key=self.__target_key_by_stage[stage]["G"])

                    self._on_g_batch_end_callbacks(progress_bar=progress_bar,
                                                   epoch=epoch,
                                                   n_epochs=self.__n_epochs,
                                                   stage=stage,
                                                   batch_i=batch_i)

                    self._on_batch_end_callbacks(progress_bar=progress_bar,
                                                 epoch=epoch,
                                                 n_epochs=self.__n_epochs,
                                                 stage=stage,
                                                 batch_i=batch_i)

                self._on_epoch_end_callbacks(epoch=epoch, n_epochs=self.__n_epochs, stage=stage)
