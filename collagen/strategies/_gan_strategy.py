from collagen.core import Callback, Trainer, Session, Module
from collagen.data import DataProvider
from collagen.data.utils import to_tuple
import torch.nn as nn
from torch.optim import Optimizer
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

    def __init__(self,
                 data_provider: DataProvider,
                 num_samples_dict: dict,
                 g_loader_names: Tuple[str] or str, d_loader_names: Tuple[str] or str,
                 g_criterion: nn.Module, d_criterion: nn.Module,
                 g_model: Module, d_model: Module,
                 g_optimizer: Optimizer, d_optimizer: Optimizer,
                 g_data_key: str, d_data_key: str,
                 g_target_key: Tuple[str] or str = (), d_target_key: Tuple[str] or str = (),
                 g_callbacks: Tuple[Callback] or Callback = (),
                 d_callbacks: Tuple[Callback] or Callback = (),
                 callbacks: Tuple[Callback] or Callback = (),
                 n_epochs: int or None = 100,
                 device: str or None = "cuda"):

        self.__data_provider = data_provider
        self.__num_samples_dict = num_samples_dict
        self.__n_epochs = n_epochs
        self.__callbacks = to_tuple(callbacks)

        # Discriminator
        self.__discriminator = dict()
        self.__discriminator["loader_names"] = to_tuple(d_loader_names)
        self.__discriminator["data_key"] = to_tuple(d_data_key)
        self.__discriminator["target_key"] = to_tuple(d_target_key)
        self.__discriminator["criterion"] = d_criterion
        self.__discriminator["optimizer"] = d_optimizer
        self.__discriminator["model"] = d_model
        self.__discriminator["callbacks"] = to_tuple(d_callbacks)

        # Generator
        self.__generator = dict()
        self.__generator["loader_names"] = to_tuple(g_loader_names)
        self.__generator["data_key"] = to_tuple(g_data_key)
        self.__generator["target_key"] = to_tuple(g_target_key)
        self.__generator["criterion"] = g_criterion
        self.__generator["optimizer"] = g_optimizer
        self.__generator["model"] = g_model
        self.__generator["callbacks"] = to_tuple(g_callbacks)

        # Discriminator
        if len(self.__discriminator["loader_names"]) != len(self.__discriminator["data_key"]):
            raise ValueError(
                "In discriminator, the number of loaders and the number of data keys must be matched."
                "({} vs {})".format(len(self.__discriminator["loader_names"]), len(self.__discriminator["data_key"])))

        # Generator
        if len(self.__generator["loader_names"]) != len(self.__generator["data_key"]):
            raise ValueError("In generator, the number of loaders and the number of sample quantities must be matched."
                             "({} vs {})".format(len(self.__generator["loader_names"]),
                                                 len(self.__generator["data_key"])))

        self.__num_batches = -1

        # TODO: get union of dicriminator's and generator's loader names
        for i, d_loader_name in enumerate(self.__discriminator["loader_names"]):
            self.__num_batches = max(len(self.__data_provider.get_loader_by_name(d_loader_name)), self.__num_batches)

        self.__use_cuda = torch.cuda.is_available() and device == "cuda"
        self.__device = torch.device("cuda" if self.__use_cuda and torch.cuda.is_available() else "cpu")

        self.__discriminator["model"].to(self.__device)
        self.__generator["model"].to(self.__device)
        self.__discriminator["criterion"].to(self.__device)
        self.__generator["criterion"].to(self.__device)

        self.__discriminator["session"] = Session(module=self.__discriminator["model"],
                                                  optimizer=self.__discriminator["optimizer"],
                                                  loss=self.__discriminator["criterion"])

        self.__generator["session"] = Session(module=self.__generator["model"],
                                              optimizer=self.__generator["optimizer"],
                                              loss=self.__generator["criterion"])

        self.__discriminator["trainer"] = Trainer(data_provider=self.__data_provider,
                                                  train_loader_names=self.__discriminator["loader_names"],
                                                  session=self.__discriminator["session"],
                                                  train_callbacks=self.__discriminator["callbacks"])

        self.__generator["trainer"] = Trainer(data_provider=self.__data_provider,
                                              train_loader_names=self.__generator["loader_names"],
                                              session=self.__generator["session"],
                                              train_callbacks=self.__generator["callbacks"])

    def _on_batch_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar):
        for cb in self.__generator["callbacks"]:
            cb.on_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage="generate",
                              strategy=self)

        for cb in self.__discriminator["callbacks"]:
            cb.on_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage="discriminate",
                              strategy=self)

        for cb in self.__callbacks:
            cb.on_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage=stage,
                              strategy=self)

    def _on_batch_end_callbacks(self, epoch, stage, n_epochs, batch_i, progress_bar):
        for cb in self.__generator["callbacks"]:
            cb.on_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage="generate",
                            strategy=self)

        for cb in self.__discriminator["callbacks"]:
            cb.on_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage="discriminate",
                            strategy=self)

        for cb in self.__callbacks:
            cb.on_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage=stage,
                            strategy=self)

    def _on_epoch_begin_callbacks(self, epoch, n_epochs, stage):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              stage="generate",
                              strategy=self)

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              stage="discriminate",
                              strategy=self)

        for cb in self.__callbacks:
            cb.on_epoch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              stage=stage,
                              strategy=self)

    def _on_epoch_end_callbacks(self, epoch, stage, n_epochs):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            stage="generate",
                            strategy=self)

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            stage="discriminate",
                            strategy=self)

        for cb in self.__callbacks:
            cb.on_epoch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            stage=stage,
                            strategy=self)

    def _on_sample_begin_callbacks(self, epoch, n_epochs, stage, batch_i, progress_bar, **kwargs):
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               n_epochs=n_epochs,
                               batch_i=batch_i,
                               progress_bar=progress_bar,
                               stage="discriminate",
                               batch_index=batch_i,
                               strategy=self)
        for cb in self.__generator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               n_epochs=n_epochs,
                               batch_i=batch_i,
                               progress_bar=progress_bar,
                               stage="generate",
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
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_end(epoch=epoch,
                             n_epochs=n_epochs,
                             batch_i=batch_i,
                             progress_bar=progress_bar,
                             stage=stage,
                             batch_index=batch_i,
                             strategy=self)

        for cb in self.__generator["callbacks"]:
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
        for cb in self.__discriminator["callbacks"]:
            cb.on_gan_d_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage=stage,
                              batch_index=batch_i,
                              strategy=self)

        for cb in self.__generator["callbacks"]:
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
        for cb in self.__discriminator["callbacks"]:
            cb.on_gan_d_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage=stage,
                            batch_index=batch_i,
                            strategy=self)

        for cb in self.__generator["callbacks"]:
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
        for cb in self.__discriminator["callbacks"]:
            cb.on_gan_g_batch_begin(epoch=epoch,
                              n_epochs=n_epochs,
                              batch_i=batch_i,
                              progress_bar=progress_bar,
                              stage=stage,
                              batch_index=batch_i,
                              strategy=self)

        for cb in self.__generator["callbacks"]:
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
        for cb in self.__discriminator["callbacks"]:
            cb.on_gan_g_batch_end(epoch=epoch,
                            n_epochs=n_epochs,
                            batch_i=batch_i,
                            progress_bar=progress_bar,
                            stage=stage,
                            batch_index=batch_i,
                            strategy=self)

        for cb in self.__generator["callbacks"]:
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
            return self.__discriminator["callbacks"]
        elif name == "G":
            return self.__generator["callbacks"]
        elif name == "minibatch":
            return self.__discriminator["callbacks"] + self.__generator["callbacks"]
        elif name == "batch":
            return self.__callbacks
        elif name == "all":
            return self.__discriminator["callbacks"] + self.__generator["callbacks"] + self.__callbacks

    def run(self):
        stage = "train"
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
                self.__data_provider.sample(**self.__num_samples_dict)
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
                getattr(self.__discriminator["trainer"], stage)(data_key=self.__discriminator["data_key"],
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
                getattr(self.__generator["trainer"], stage)(data_key=self.__generator["data_key"],
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
