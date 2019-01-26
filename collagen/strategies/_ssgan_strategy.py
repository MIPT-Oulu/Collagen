from collagen.core import Trainer, Session, Module
from collagen.core import Callback
from collagen.data import DataProvider
from collagen.data.utils import to_tuple
import torch.nn as nn
from torch.optim import Optimizer
from typing import Tuple
import torch
from tqdm import tqdm


class SSGANStrategy(object):
    """
        Implements a part of the training loop by passing the available batches through the model.

        Parameters
        ----------
        data_provider : DataProvider
            Data provider. Controlled outside and samples mini-batches.
    """

    def __init__(self, data_provider: DataProvider,
                 train_samples_dict: dict,
                 eval_samples_dict: dict,
                 d_trainer: Trainer, g_trainer: Trainer,
                 n_epochs: int or None = 100,
                 callbacks: Tuple[Callback] or Callback = None,
                 device: str or None = "cuda"):

        self.__stage_names = {"train", "eval"}
        self.__model_names = ("G", "D")
        self.__n_epochs = n_epochs
        self.__callbacks = to_tuple(callbacks)
        self.__data_provider = data_provider

        self.__num_samples_by_name = {"train": train_samples_dict, "eval": eval_samples_dict}

        # TODO: get union of dicriminator's and generator's loader names
        self.__num_batches = -1

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")
        self.__trainers = {"D": d_trainer, "G": g_trainer}

    def _on_batch_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar):
        for model_name in self.__model_names:
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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
            for cb in getattr(self.__trainers[model_name], f'get_{stage}_callbacks')():
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

    def get_callbacks_by_name(self, name):
        if name == "D":
            return self.__trainers[name].get_train_callbacks()
        elif name == "G":
            return self.__trainers[name].get_train_callbacks()
        elif name == "minibatch":
            return self.__trainers["D"].get_train_callbacks() + self.__trainers["G"].get_train_callbacks()
        elif name == "batch":
            return self.__callbacks
        elif name == "all":
            return self.__trainers["D"].get_train_callbacks() \
                   + self.__trainers["G"].get_train_callbacks() \
                   + self.__callbacks

    def run(self):
        for stage in self.__stage_names:
            for epoch in range(self.__n_epochs):
                self._on_epoch_begin_callbacks(epoch=epoch, stage=stage, n_epochs=self.__n_epochs)

                progress_bar = tqdm(range(self.__num_batches), total=self.__num_batches,
                                    desc=f'Epoch [{epoch}]::')
                for batch_i in progress_bar:
                    self._on_sample_begin_callbacks(progress_bar=progress_bar,
                                                    epoch=epoch,
                                                    n_epochs=self.__n_epochs,
                                                    stage=stage,
                                                    batch_i=batch_i)
                    self.__data_provider.sample(**self.__num_samples_by_name[stage])
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
                    getattr(self.__trainers["D"], stage)(data_key=self.__discriminator["data_key"],
                                                                    target_key=self.__discriminator["target_key"])
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
                    getattr(self.__trainers["G"], stage)(data_key=self.__generator["data_key"],
                                                                target_key=self.__generator["target_key"])
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
