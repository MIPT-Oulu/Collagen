from ..data import DataProvider
from ._session import Session
from typing import Tuple, Dict
from ..metrics._meters import BaseMeter, RunningAverageMeter


class BaseScheduler(object):
    def __init__(self, session: Tuple[Session] or Session, steps: Tuple[int], meter: BaseMeter):
        self.__session: Tuple[Session] or Session = session
        self.__counter = 0
        self.__steps: Tuple[int] = steps
        self.__meter: BaseMeter = meter

    def step(self):
        raise NotImplementedError


class BaseTrainStrategy(object):
    """
    Implements a part of the training loop by passing the available batches through the model.

    Parameters
    ----------
    data_provider : DataProvider
        Data provider. Controlled outside and samples mini-batches.
    train_loader_name : str
        Name of the training loader, which is a part of DataProvider.
    val_loader_name : str
        Name of the val loader, which is a part of DataProvider.
    session : Session
        Session to operate with
    train_schedulers : Tuple[BaseScheduler]
        Schedulers, which allow to adjust the session parameters during training.
        This can be useful for implementing super-convergence and stochastic weight averaging.
    train_meters : Tuple[RunningAverageMeter] or RunningAverageMeter
        Meters batch-wise, which track losses / metrics during training.
    val_meters : Tuple[BaseMeter] or BaseMeter
        Meters batch-wise, which track losses / metrics during validation.
    """
    def __init__(self, data_provider: DataProvider,
                 train_loader_name: str,
                 val_loader_name: str,
                 session: Session,
                 train_schedulers: Tuple[BaseScheduler] or None,
                 train_meters: Tuple[RunningAverageMeter] or RunningAverageMeter,
                 val_meters: Tuple[BaseMeter] or BaseMeter):

        self.__data_provider: DataProvider = data_provider
        self.__session: Session = session
        self.__train_loader_name: str = train_loader_name
        self.__val_loader_name: str = val_loader_name
        self.__train_schedulers: Tuple[BaseScheduler] = train_schedulers
        self.__train_meters: Tuple[RunningAverageMeter] or RunningAverageMeter = train_meters

        if not isinstance(train_meters, Tuple):
            self.__train_meters: Tuple[RunningAverageMeter] or RunningAverageMeter = (train_meters, )

        self.__val_meters: Tuple[BaseMeter] or BaseMeter = val_meters
        if not isinstance(val_meters, tuple):
            self.__val_meters: Tuple[BaseMeter] = (val_meters, )

    def train(self, accumulate_grad=False):
        cur_loader_state = self.__data_provider.state_dict()[self.__train_loader_name]
        n_iter = len(cur_loader_state["samples"])

        i = 0
        for i in range(n_iter-1):
            batch = cur_loader_state["samples"][i]
            train_result = self.__session.train_step(batch, accumulate_grad, True)
            for meter in self.__train_meters:
                meter.update(train_result)

            if self.__train_schedulers is not None:
                for scheduler in self.__train_schedulers:
                    scheduler.step()

        batch = cur_loader_state["samples"][i]
        train_result = self.__session.train_step(batch, False, True)
        for meter in self.__train_meters:
            meter.update(train_result)

    def eval(self):
        cur_loader_state = self.__data_provider.state_dict()[self.__val_loader_name]
        n_iter = len(cur_loader_state["samples"])

        if n_iter != 1:
            raise ValueError(f"Number of validation batches drawn from DataProvider must be 1, "
                             f"but found {n_iter}")

        batch = cur_loader_state["samples"][0]
        eval_res = self.__session.eval_step(batch, True)
        for meter in self.__val_meters:
            meter.update(eval_res)
