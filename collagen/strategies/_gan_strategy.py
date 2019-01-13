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
                 g_data_provider: DataProvider, d_data_provider: DataProvider,
                 g_loader_names: Tuple[str] or str, d_loader_names: Tuple[str] or str,
                 g_criterion: nn.Module, d_criterion: nn.Module,
                 g_model: Module, d_model: Module,
                 g_optimizer: Optimizer, d_optimizer: Optimizer,
                 g_data_key: str, d_data_key: str,
                 g_target_key: str = (), d_target_key: str = (),
                 g_num_samples: Tuple[int] or int or None = (), d_num_samples: Tuple[int] or int or None = (),
                 g_callbacks: Tuple[Callback] or Callback = (),
                 d_callbacks: Tuple[Callback] or Callback = (),
                 n_epochs: int or None = 100,
                 device: str or None = "cuda"):
        self.__discriminator = dict()
        self.__discriminator["data_provider"] = d_data_provider
        self.__discriminator["loader_names"] = unify_tuple(d_loader_names)
        self.__discriminator["data_key"] = d_data_key
        self.__discriminator["target_key"] = d_target_key
        self.__discriminator["num_samples"] = unify_tuple(d_num_samples)
        self.__discriminator["criterion"] = d_criterion
        self.__discriminator["optimizer"] = d_optimizer
        self.__discriminator["model"] = d_model
        self.__discriminator["callbacks"] = d_callbacks

        self.__generator = dict()
        self.__generator["data_provider"] = g_data_provider
        self.__generator["loader_names"] = unify_tuple(g_loader_names)
        self.__generator["data_key"] = g_data_key
        self.__generator["target_key"] = g_target_key
        self.__generator["num_samples"] = unify_tuple(g_num_samples)
        self.__generator["criterion"] = g_criterion
        self.__generator["optimizer"] = g_optimizer
        self.__generator["model"] = g_model
        self.__generator["callbacks"] = g_callbacks

        self.__n_epochs = n_epochs

        # Discriminator
        if len(self.__discriminator["loader_names"]) != len(self.__discriminator["num_samples"]):
            raise ValueError(
                "In discriminator, the number of loaders and the number of sample quantities must be matched."
                "({} vs {})".format(len(self.__discriminator["loader_names"]),
                                    len(self.__discriminator["num_samples"])))

        # Generator
        if len(self.__generator["loader_names"]) != len(self.__generator["num_samples"]):
            raise ValueError("In generator, the number of loaders and the number of sample quantities must be matched."
                             "({} vs {})".format(len(self.__generator["loader_names"]),
                                                 len(self.__generator["num_samples"])))

        self.__discriminator["sampling_kwargs"] = dict()
        self.__generator["sampling_kwargs"] = dict()
        self.__num_batches = -1
        for i, d_loader_name in enumerate(self.__discriminator["loader_names"]):
            self.__num_batches = max(len(self.__discriminator["data_provider"].get_loader_by_name(d_loader_name)), self.__num_batches)
            self.__discriminator["sampling_kwargs"][d_loader_name] = self.__discriminator["num_samples"][i]

        for i, g_loader_name in enumerate(self.__generator["loader_names"]):
            self.__generator["sampling_kwargs"][g_loader_name] = self.__generator["num_samples"][i]

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

        self.__discriminator["trainer"] = Trainer(data_provider=self.__discriminator["data_provider"],
                                                  train_loader_names=self.__discriminator["loader_names"],
                                                  session=self.__discriminator["session"],
                                                  train_callbacks=self.__discriminator["callbacks"])

        self.__generator["trainer"] = Trainer(data_provider=self.__generator["data_provider"],
                                              train_loader_names=self.__generator["loader_names"],
                                              session=self.__generator["session"],
                                              train_callbacks=self.__generator["callbacks"])

    def _on_epoch_begin_callbacks(self, epoch):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              stage="generate",
                              data_provider=self.__generator["data_provider"],
                              data_key=self.__generator["data_key"],
                              target_key=self.__generator["target_key"],
                              session=self.__generator["session"])

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_begin(epoch=epoch,
                              stage="discriminate",
                              data_provider=self.__discriminator["data_provider"],
                              data_key=self.__discriminator["data_key"],
                              target_key=self.__discriminator["target_key"],
                              session=self.__discriminator["session"])

    def _on_epoch_end_callbacks(self, epoch):
        for cb in self.__generator["callbacks"]:
            cb.on_epoch_end(stage="generate",
                            data_provider=self.__generator["data_provider"],
                            data_key=self.__generator["data_key"],
                            target_key=self.__generator["target_key"],
                            session=self.__generator["session"])

        for cb in self.__discriminator["callbacks"]:
            cb.on_epoch_end(stage="discriminate",
                            data_provider=self.__discriminator["data_provider"],
                            data_key=self.__discriminator["data_key"],
                            target_key=self.__discriminator["target_key"],
                            session=self.__discriminator["session"])

    def _on_sample_begin_callbacks(self, epoch, stage, batch_i):
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               stage=stage,
                               batch_index=batch_i,
                               data_provider=self.__discriminator["data_provider"],
                               data_key=self.__discriminator["data_key"],
                               target_key=self.__discriminator["target_key"],
                               session=self.__discriminator["session"])
        for cb in self.__generator["callbacks"]:
            cb.on_sample_begin(epoch=epoch,
                               stage=stage,
                               batch_index=batch_i,
                               data_provider=self.__generator["data_provider"],
                               data_key=self.__generator["data_key"],
                               target_key=self.__generator["target_key"],
                               session=self.__generator["session"])

    def _on_sample_end_callbacks(self,epoch, stage, batch_i):
        for cb in self.__discriminator["callbacks"]:
            cb.on_sample_end(epoch=epoch,
                             stage=stage,
                             batch_index=batch_i,
                             data_provider=self.__discriminator["data_provider"],
                             data_key=self.__discriminator["data_key"],
                             target_key=self.__discriminator["target_key"],
                             session=self.__discriminator["session"])

        for cb in self.__generator["callbacks"]:
            cb.on_sample_end(epoch=epoch,
                             stage=stage,
                             batch_index=batch_i,
                             data_provider=self.__generator["data_provider"],
                             data_key=self.__generator["data_key"],
                             target_key=self.__generator["target_key"],
                             session=self.__generator["session"])

    def run(self):
        stage = "train"
        for epoch in range(self.__n_epochs):
            self._on_epoch_begin_callbacks(epoch=epoch)
            for batch_i in tqdm(range(self.__num_batches), total=self.__num_batches, desc=f'Epoch [{epoch}] ::'): #range(self.__num_batches):
                self._on_sample_begin_callbacks(epoch=epoch, stage="train", batch_i=batch_i)
                self.__generator["data_provider"].sample(**self.__generator["sampling_kwargs"])
                self.__discriminator["data_provider"].sample(**self.__discriminator["sampling_kwargs"])
                self._on_sample_end_callbacks(epoch=epoch, stage="train", batch_i=batch_i)
                getattr(self.__generator["trainer"], stage)(data_key=self.__generator["data_key"],
                                                            target_key=self.__generator["target_key"])
                getattr(self.__discriminator["trainer"], stage)(data_key=self.__discriminator["data_key"],
                                                                target_key=self.__discriminator["target_key"])
            self._on_epoch_end_callbacks(epoch=epoch)
