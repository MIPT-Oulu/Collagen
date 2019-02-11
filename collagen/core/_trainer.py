from ..data import DataProvider
from ._session import Session
from typing import Tuple
from collagen.core._callback import Callback
from ..data.utils import cast_tensor


class Trainer(object):
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

    def __init__(self, data_provider: DataProvider,
                 train_loader_names: str or Tuple[str] or None,
                 session: Session,
                 val_loader_names: str or Tuple[str] = None,
                 train_callbacks: Tuple[Callback] or Callback or None = None,
                 val_callbacks: Tuple[Callback] or Callback or None = None):

        if train_callbacks is None:
            train_callbacks = ()
        if val_callbacks is None:
            val_callbacks = ()

        self.__data_provider: DataProvider = data_provider
        self.__session: Session = session

        self.__train_loader_names: str or Tuple[str] = train_loader_names
        if isinstance(self.__train_loader_names, str):
            self.__train_loader_names = (self.__train_loader_names,)

        self.__val_loader_names: str or Tuple[str] = val_loader_names
        if isinstance(self.__val_loader_names, str):
            self.__val_loader_names = (self.__val_loader_names,)

        self.__train_callbacks: Tuple[Callback] or Callback = train_callbacks
        self.__val_callbacks: Tuple[Callback] or Callback = val_callbacks

        if not isinstance(val_callbacks, tuple):
            self.__val_callbacks: Tuple[Callback] or Callback = (val_callbacks,)

        if not isinstance(train_callbacks, tuple):
            self.__train_callbacks: Tuple[Callback] = (train_callbacks,)

        self.__train_batches_count = 0
        self.__eval_batches_count = 0

    def train(self, data_key: Tuple[str] or str = 'img',
              target_key: Tuple[str] or str = 'target',
              accumulate_grad=False, cast_target=None):
        """
        Runs session in train mode as many iterations as given in the train loader.

        This method does not return anything a stores everything in the callbacks.

        Parameters
        ----------
        data_key : Tuple[str] or str
            Key of the dictionary, which corresponds to the data. Sometimes (e.g. in Siamese models),
            we need two items thus we might need multiple keys.
        target_key : Tuple[str] or str
            Key of the dictionary, which corresponds to the target. In case of models with e.g. multiple
            heads and heterogeneous outputs, it could be useful to use multiple keys.
        accumulate_grad : bool
            Whether to accumulate gradient.
        cast_target : None or str
            Performs type casting for target

        """
        for ind, loader_name in enumerate(self.__train_loader_names):
            cur_loader_state = self.__data_provider.state_dict()[loader_name]
            n_iter = len(cur_loader_state["samples"])

            if isinstance(data_key, str):
                data_key = (data_key,)

            if isinstance(target_key, str):
                target_key = (target_key,)

            i = 0
            for i in range(n_iter - 1):
                batch = cur_loader_state["samples"][i]
                for cb in self.__train_callbacks:
                    cb.on_minibatch_begin(loader_name=loader_name,
                                          batches_count=self.__train_batches_count,
                                          batch=batch,
                                          data_key=data_key[ind],
                                          target_key=target_key,
                                          session=self.__session)

                input_data = batch[data_key[ind]]
                # target = tuple([cast_tensor(batch[key_i], cast_target) for key_i in target_key])
                target = cast_tensor(batch[target_key[ind]], cast_target)
                loss, train_result = self.__session.train_step(input_data,
                                                               target,
                                                               accumulate_grad=accumulate_grad,
                                                               return_out=True, callbacks=self.__train_callbacks)
                self.__train_batches_count += 1

                for cb in self.__train_callbacks:
                    cb.on_minibatch_end(loader_name=loader_name,
                                        batches_count=self.__train_batches_count,
                                        loss=loss,
                                        input=input_data,
                                        output=train_result,
                                        target=target,
                                        data_key=data_key[ind],
                                        target_key=target_key[ind],
                                        session=self.__session)

            batch = cur_loader_state["samples"][i]
            for cb in self.__train_callbacks:
                cb.on_minibatch_begin(loader_name=loader_name,
                                      batches_count=self.__train_batches_count,
                                      batch=batch,
                                      data_key=data_key[ind],
                                      target_key=target_key,
                                      session=self.__session)

            input_data = batch[data_key[ind]]
            # target = tuple([cast_tensor(batch[key_i], cast_target) for key_i in target_key])
            target = cast_tensor(batch[target_key[ind]], cast_target)
            loss, train_result = self.__session.train_step(input_data,
                                                           target,
                                                           accumulate_grad=False,
                                                           return_out=True,
                                                           callbacks=self.__train_callbacks)
            self.__train_batches_count += 1
            # TODO: support tuple of target_key, inputs and targets
            for cb in self.__train_callbacks:
                cb.on_minibatch_end(loader_name=loader_name,
                                    batches_count=self.__train_batches_count,
                                    loss=loss,
                                    input=input_data,
                                    output=train_result,
                                    target=target,
                                    data_key=data_key[ind],
                                    target_key=target_key[ind],
                                    session=self.__session)

    def eval(self, data_key: Tuple[str] or str = 'img',
             target_key: Tuple[str] or str = 'target', cast_target=None):
        """
        Runs session in `eval` mode as many iterations as given in the validation / test loader.

        This method does not return anything a stores everything in the callbacks.
        The callbacks here are called before the minibatch and after minibatch

        Parameters
        ----------
        data_key : Tuple[str] or str
            Key of the dictionary, which corresponds to the data. Sometimes (e.g. in Siamese models),
            we need two items thus we might need multiple keys.
        target_key : Tuple[str] or str
            Key of the dictionary, which corresponds to the target. In case of models with e.g. multiple
            heads and heterogeneous outputs, it could be useful to use multiple keys.
        cast_target : None or str
            Performs type casting for target

        """

        for ind, loader_name in enumerate(self.__val_loader_names):
            cur_loader_state = self.__data_provider.state_dict()[loader_name]
            n_iter = len(cur_loader_state["samples"])

            if isinstance(data_key, str):
                data_key = (data_key,)

            if isinstance(target_key, str):
                target_key = (target_key,)

            i = 0
            for i in range(n_iter - 1):
                batch = cur_loader_state["samples"][i]
                for cb in self.__val_callbacks:
                    cb.on_minibatch_begin(loader_name=loader_name,
                                          batches_count=self.__eval_batches_count,
                                          batch=batch,
                                          data_key=data_key[ind],
                                          target_key=target_key,
                                          session=self.__session)

                input_data = batch[data_key[ind]]
                target = tuple([cast_tensor(batch[key_i], cast_target) for key_i in target_key])
                loss, eval_result = self.__session.eval_step(input_data,
                                                             target[ind],
                                                             return_out=True,
                                                             callbacks=self.__val_callbacks)
                self.__eval_batches_count += 1

                for cb in self.__val_callbacks:
                    cb.on_minibatch_end(loader_name=loader_name,
                                        batches_count=self.__eval_batches_count,
                                        loss=loss,
                                        input=input_data,
                                        output=eval_result,
                                        target=target[ind],
                                        data_key=data_key[ind],
                                        target_key=target_key[ind],
                                        session=self.__session)

            batch = cur_loader_state["samples"][i]
            for cb in self.__val_callbacks:
                cb.on_minibatch_begin(loader_name=loader_name,
                                      batches_count=self.__eval_batches_count,
                                      batch=batch,
                                      data_key=data_key[ind],
                                      target_key=target_key,
                                      session=self.__session)

            input_data = batch[data_key[ind]]
            target = tuple([cast_tensor(batch[key_i], cast_target) for key_i in target_key])
            # target = cast_tensor(batch[key_i], cast_target)
            loss, eval_result = self.__session.eval_step(input_data,
                                                         target[ind],
                                                         return_out=True,
                                                         callbacks=self.__val_callbacks)
            self.__eval_batches_count += 1

            for cb in self.__val_callbacks:
                cb.on_minibatch_end(loader_name=loader_name,
                                    batches_count=self.__eval_batches_count,
                                    loss=loss,
                                    input=input_data,
                                    output=eval_result,
                                    target=target[ind],
                                    data_key=data_key[ind],
                                    target_key=target_key[ind],
                                    session=self.__session)

    def get_callbacks_by_stage(self, stage):
        if stage == "train" or "train" in stage:
            return self.__train_callbacks
        elif stage == "eval" or "eval" in stage:
            return self.__val_callbacks
        elif stage is None:
            return self.__train_callbacks + self.__val_callbacks
        else:
            raise ValueError("stage must be `train`, `eval`, tuple of both or None, but found {}".format(stage))
