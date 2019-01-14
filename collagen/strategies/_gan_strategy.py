from collagen.core import Callback, Trainer, Session, Module
from collagen.data import DataProvider
from collagen.data.utils import unify_tuple
import torch.nn as nn
from torch.optim import Optimizer
import pandas as pd
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
                 g_target_key: str = (), d_target_key: str = (),
                 g_callbacks: Tuple[Callback] or Callback = (),
                 d_callbacks: Tuple[Callback] or Callback = (),
                 n_epochs: int or None = 100,
                 device: str or None = "cuda"):
        self.__discriminator = dict()
        self.__data_provider = data_provider
        self.__num_samples_dict = num_samples_dict
        self.__n_epochs = n_epochs

        # Discriminator
        self.__discriminator["loader_names"] = unify_tuple(d_loader_names)
        self.__discriminator["data_key"] = unify_tuple(d_data_key)
        self.__discriminator["target_key"] = unify_tuple(d_target_key)
        self.__discriminator["criterion"] = d_criterion
        self.__discriminator["optimizer"] = d_optimizer
        self.__discriminator["model"] = d_model
        self.__discriminator["callbacks"] = d_callbacks

        # Generator
        self.__generator = dict()
        self.__generator["loader_names"] = unify_tuple(g_loader_names)
        self.__generator["data_key"] = unify_tuple(g_data_key)
        self.__generator["target_key"] = unify_tuple(g_target_key)
        self.__generator["criterion"] = g_criterion
        self.__generator["optimizer"] = g_optimizer
        self.__generator["model"] = g_model
        self.__generator["callbacks"] = g_callbacks

        # Discriminator
        if len(self.__discriminator["loader_names"]) != len(self.__discriminator["data_key"]):
            raise ValueError(
                "In discriminator, the number of loaders and the number of data keys must be matched."
                "({} vs {})".format(len(self.__discriminator["loader_names"]), len(self.__discriminator["data_key"])))

        # Generator
        if len(self.__generator["loader_names"]) != len(self.__generator["data_key"]):
            raise ValueError("In generator, the number of loaders and the number of sample quantities must be matched."
                             "({} vs {})".format(len(self.__generator["loader_names"]), len(self.__generator["data_key"])))

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

    def _on_epoch_begin_callbacks(self, epoch):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              stage="generate",
                              data_provider=self.__data_provider,
                              data_key=self.__generator["data_key"],
                              target_key=self.__generator["target_key"],
                              session=self.__generator["session"])

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              stage="discriminate",
                              data_provider=self.__data_provider,
                              data_key=self.__discriminator["data_key"],
                              target_key=self.__discriminator["target_key"],
                              session=self.__discriminator["session"])

    def _on_epoch_end_callbacks(self, epoch):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_end(stage="generate",
                            epoch=epoch,
                            num_epochs=self.__n_epochs,
                            data_provider=self.__data_provider,
                            data_key=self.__generator["data_key"],
                            target_key=self.__generator["target_key"],
                            session=self.__generator["session"])

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_end(stage="discriminate",
                            epoch=epoch,
                            num_epochs=self.__n_epochs,
                            data_provider=self.__data_provider,
                            data_key=self.__discriminator["data_key"],
                            target_key=self.__discriminator["target_key"],
                            session=self.__discriminator["session"])

    def _on_sample_begin_callbacks(self, epoch, stage, batch_i):
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               stage=stage,
                               batch_index=batch_i,
                               data_provider=self.__data_provider,
                               data_key=self.__discriminator["data_key"],
                               target_key=self.__discriminator["target_key"],
                               session=self.__discriminator["session"])
        for cb in self.__generator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               stage=stage,
                               batch_index=batch_i,
                               data_provider=self.__data_provider,
                               data_key=self.__generator["data_key"],
                               target_key=self.__generator["target_key"],
                               session=self.__generator["session"])

    def _on_sample_end_callbacks(self,epoch, stage, batch_i):
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_end(epoch=epoch,
                             stage=stage,
                             batch_index=batch_i,
                             data_provider=self.__data_provider,
                             data_key=self.__discriminator["data_key"],
                             target_key=self.__discriminator["target_key"],
                             session=self.__discriminator["session"])

        for cb in self.__generator["callbacks"]:
            cb.on_sample_end(epoch=epoch,
                             stage=stage,
                             batch_index=batch_i,
                             data_provider=self.__data_provider,
                             data_key=self.__generator["data_key"],
                             target_key=self.__generator["target_key"],
                             session=self.__generator["session"])

    def run(self):
        stage = "train"
        for epoch in range(self.__n_epochs):
            self._on_epoch_begin_callbacks(epoch=epoch)
            metrics_desc = ""
            progress_bar = tqdm(range(self.__num_batches), total=self.__num_batches, desc=f'Epoch [{epoch}]::{metrics_desc}')
            for batch_i in progress_bar:
                self._on_sample_begin_callbacks(epoch=epoch, stage="train", batch_i=batch_i)
                self.__data_provider.sample(**self.__num_samples_dict)
                self._on_sample_end_callbacks(epoch=epoch, stage="train", batch_i=batch_i)
                getattr(self.__generator["trainer"], stage)(data_key=self.__generator["data_key"],
                                                            target_key=self.__generator["target_key"])
                getattr(self.__discriminator["trainer"], stage)(data_key=self.__discriminator["data_key"],
                                                                target_key=self.__discriminator["target_key"])
                list_metrics_desc = [str(cb) for cb in self.__generator["callbacks"]]
                list_metrics_desc += [str(cb) for cb in self.__discriminator["callbacks"]]
                metrics_desc = ", ".join(list_metrics_desc)
                progress_bar.set_description(f'Epoch [{epoch}]::{metrics_desc}')
            self._on_epoch_end_callbacks(epoch=epoch)
