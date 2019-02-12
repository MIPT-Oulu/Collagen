import torch.nn as nn
from os.path import exists, join
from datetime import datetime
from os import mkdir
from typing import Tuple
import torch
from collagen.core import Callback
from collagen.data.utils import to_tuple


class ModelSaver(Callback):
    def __init__(self, metric_names: Tuple[str] or str, conditions: Tuple[str] or str, model: nn.Module,
                 save_dir: str, prefix: str = ""):
        super().__init__(type="saver")
        self.__metric_names = to_tuple(metric_names)
        self.__conditions = to_tuple(conditions)
        self.__prefix = prefix
        self.__save_dir = save_dir

        if not exists(self.__save_dir):
            print("Not found directory {} to save models. Create the directory.".format(self.__save_dir))
            mkdir(self.__save_dir)

        if len(self.__metric_names) != len(self.__conditions):
            raise ValueError("Metric names ({}) and conditions ({}) must be the same, "
                             "but got {} != {}".format(len(self.__metric_names),
                                                       len(self.__conditions),
                                                       len(self.__metric_names),
                                                       len(self.__conditions)))

        self.__best_metrics = dict()
        for i, metric_name in enumerate(metric_names):
            cond = self.__conditions[i].lower()
            if cond in ["min", "max"]:
                self.__best_metrics[metric_name] = dict()
                self.__best_metrics[metric_name]["value"] = float('Inf') if cond == "min" else float('-Inf')
                self.__best_metrics[metric_name]["cond"] = cond
            else:
                raise ValueError('Values of conditions must be either min or max, but got {}'.format(cond))

        self.__model = model

    def __check_cond(self, value, metric_name):
        is_improved = False
        if self.__best_metrics[metric_name]["cond"] == "min" and self.__best_metrics[metric_name]["value"] > value:
            is_improved = True
        elif self.__best_metrics[metric_name]["cond"] == "max" and self.__best_metrics[metric_name]["value"] < value:
            is_improved = True

        return is_improved

    def on_epoch_end(self, epoch, stage, strategy, **kwargs):
        improved_metrics = dict()
        for cb in strategy.get_callbacks_by_name("minibatch", stage=stage):
            cb_name = str(cb)
            if cb.get_type() == "meter" and cb_name in self.__best_metrics:
                cb_value = cb.current()
                if self.__check_cond(value=cb_value, metric_name=cb_name):
                    improved_metrics[cb_name] = cb_value

        if len(improved_metrics) == len(self.__best_metrics):
            list_metrics = []
            for metric_name in self.__best_metrics:
                self.__best_metrics[metric_name]["value"] = improved_metrics[metric_name]
                list_metrics += [metric_name.replace('/', ''), "{0:.3f}".format(improved_metrics[metric_name])]
            metrics_desc = "_".join(list_metrics)
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = "_".join([self.__prefix, date_time, metrics_desc]) + ".pth"
            model_fullname = join(self.__save_dir, model_name)
            torch.save(self.__model.state_dict(), model_fullname)

