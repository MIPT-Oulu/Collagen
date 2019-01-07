from collagen.core import Callback, Trainer, Session, Module
from collagen.data import DataProvider, ItemLoader, Splitter
import torch.nn as nn
from torch.optim import Optimizer
import pandas as pd
import torch
import tqdm


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

    def __init__(self, data_frame: pd.DataFrame,
                 args: dict,
                 splitter: Splitter,
                 loss: nn.Module,
                 model: Module,
                 optimizer: Optimizer,
                 transform,
                 parse_item_cb,
                 train_callbacks: Tuple[Callback] or Callback = None,
                 val_callbacks: Tuple[Callback] or Callback = None,
                 device: str = "cuda"):
        self.__data_frame: pd.DataFrame = data_frame
        self.__splitter: Splitter = splitter
        self.__loss: nn.Module = loss
        self.__optimizer: Optimizer = optimizer
        self.__model: Module = model

        self.__train_callbacks: Tuple[Callback] or Callback = train_callbacks
        self.__val_callbacks: Tuple[Callback] or Callback = val_callbacks
        if not isinstance(val_callbacks, tuple):
            self.__val_callbacks: Tuple[Callback] or Callback = (val_callbacks, )

        if not isinstance(train_callbacks, tuple):
            self.__train_callbacks: Tuple[Callback] = (train_callbacks, )

        self.__args: dict = args

        self.__parse_item_callback = parse_item_cb
        self.__transform = transform
        self.__device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.__model.to(self.__device)
        self.__loss.to(self.__device)

    def run(self):
        splitter_kwargs = self.__args["splitter"]
        itemloader_kwargs = self.__args["itemloader"]
        # model_kwargs = self.__args["model"]
        train_kwargs = self.__args["train"]
        data_kwargs = self.__args["data"]

        for fold_id, (df_train, df_val) in enumerate(self.__splitter(self.__data_frame, **splitter_kwargs)):
            item_loaders = dict()

            for stage, df, cbs in zip(['train', 'eval'], [df_train, df_val], [self.__train_callbacks, self.__val_callbacks]):
                if isinstance(cbs, Callback):
                    cbs.on_itemloader_begin()
                else:
                    for cb in cbs:
                        cb.on_itemloader_begin()

                item_loaders[f'{fold_id}_{stage}'] = ItemLoader(meta_data=df,
                                                                transform=self.__transform,
                                                                parse_item_cb=self.__parse_item_callback.on_parse_item,
                                                                **itemloader_kwargs)

                if isinstance(cbs, Callback):
                    cbs.on_itemloader_end()
                else:
                    for cb in cbs:
                        cb.on_itemloader_end()

            data_provider = DataProvider(item_loaders)

            se = Session(module=self.__model, optimizer=self.__optimizer, loss=self.__loss)

            trainer = Trainer(data_provider, f'{fold_id}_train', f'{fold_id}_eval', se)

            for epoch in range(train_kwargs.n_epochs):
                for stage in ['train', 'eval']:
                    n_batches = len(item_loaders[f'{fold_id}_{stage}'])
                    for batch_i in tqdm(range(n_batches), desc=f'Fold [{fold_id}] | Epoch [{epoch}] | {stage}::'):
                        for cb in self.__train_callbacks:
                            cb.on_sample_begin()
                        data_provider.sample(**{f'{fold_id}_{stage}': 1})
                        for cb in self.__train_callbacks:
                            cb.on_sample_end()
                        getattr(trainer, stage)(data_key=data_kwargs["input_tag"], target_key=data_kwargs["target_tag"])

