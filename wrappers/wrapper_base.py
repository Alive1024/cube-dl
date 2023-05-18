import inspect
from typing import Iterable, Optional, Callable, Union, Dict
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from pytorch_lightning import LightningModule

try:
    from torchmetrics.metric import Metric
    TORCHMETRICS_INSTALLED = True
except ModuleNotFoundError:
    TORCHMETRICS_INSTALLED = False


class TaskWrapperBase(LightningModule, metaclass=ABCMeta):
    LOSS_FUNCTION_T = Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    METRICS_T = Optional[Union[Dict[str, Callable], Iterable[Callable], Callable]]

    def __init__(self,
                 loss_function: LOSS_FUNCTION_T,
                 validate_metrics: METRICS_T = None,
                 test_metrics: METRICS_T = None):
        super().__init__()
        self.loss_function = loss_function
        self.validate_metrics = self._process_metrics(validate_metrics, "val", loss_function)
        self.test_metrics = self._process_metrics(test_metrics, "test", loss_function)

    @staticmethod
    def _get_name_of_anything(sth) -> str:
        if inspect.isfunction(sth) or inspect.isclass(sth) or inspect.ismethod(sth):
            return sth.__name__
        else:  # general objects
            return sth.__class__.__name__

    @staticmethod
    def _generate_name_for_nameless_metric(metric_instance, prefix: str, metrics: dict):
        fixed_part = f"{prefix}_{TaskWrapperBase._get_name_of_anything(metric_instance)}"
        metric_names = metrics.keys()

        # Add number suffix to avoid name conflicts.
        cnt = 0
        while True:
            name = f"{fixed_part}_{cnt}"
            if name not in metric_names:
                return name
            cnt += 1

    @staticmethod
    def _set_torchmetrics_attrs(module, metrics: dict):
        """
        For subclasses (task wrappers) to call.
        In order to use "torchmetrics" 's metrics correctly, we must set the "torchmetrics" 's metrics
        as attributes of the `module` object.
        Note: calling this method within this class to set "torchmetrics" 's metrics as this class's
        attributes will NOT work because of the behaviour of  "PyTorch-Lightning" `LightningModule`'s `log` method.
        """
        if TORCHMETRICS_INSTALLED:
            for name, metric in metrics.items():
                if isinstance(metric, Metric):
                    setattr(module, name, metric)

    @staticmethod
    def _process_metrics(metrics: METRICS_T,
                         metric_name_prefix: str,
                         loss_function: LOSS_FUNCTION_T) -> Dict[str, Callable]:
        """
        Process metric(s) to make it/them become a dict, where the key(s) is/are provided/generated name(s),
        and the value(s) is/are Callable.
        """
        # Use the loss function as the default metric if any metric is not provided.
        if metrics is None:
            return {f"{metric_name_prefix}_{TaskWrapperBase._get_name_of_anything(loss_function)}": loss_function}

        if isinstance(metrics, dict):
            return metrics
        else:
            processed_metrics = {}

            # "torchmetrics" 's metric classes override the `__iter__` method
            # (which means it is `Iterable`), but do not really implement it,
            # only raise a `NotImplementedError` when trying to enumerate it.
            # Hence, wrap it into a list is necessary.
            if not isinstance(metrics, Iterable) or (TORCHMETRICS_INSTALLED and isinstance(metrics, Metric)):
                metrics = [metrics]

            for metric in metrics:
                generated_name = TaskWrapperBase._generate_name_for_nameless_metric(metric,
                                                                                    prefix=metric_name_prefix,
                                                                                    metrics=processed_metrics)
                processed_metrics[generated_name] = metric

        return processed_metrics

    def get_init_args(self) -> dict:
        all_init_args: inspect.Signature = inspect.signature(self.__class__.__init__)
        filtered_init_args = {}
        for key in all_init_args.parameters.keys():
            if key == "self":
                continue
            filtered_init_args[key] = getattr(self, key)
        return filtered_init_args

    @staticmethod
    @abstractmethod
    def save_predictions(predictions, save_dir):
        pass

    @abstractmethod
    def predict_from_raw_data(self, src_dir, save_dir):
        pass
