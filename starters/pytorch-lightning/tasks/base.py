import inspect
from abc import ABCMeta
from collections.abc import Callable

import pytorch_lightning as pl
import torch
from cube_dl.core import CubeTaskModule
from torch import nn
from torchmetrics import Metric, MetricCollection

LOSS_FUNCTION_T = nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
METRICS_T = dict[str, Callable | MetricCollection] | MetricCollection


class TaskBase(pl.LightningModule, CubeTaskModule, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @staticmethod
    def set_torchmetrics_attrs(module, metrics: dict):
        """Set metric objects from torchmetrics as the `module`'s attributes.

        For subclasses to call.
        In order to use "torchmetrics" 's metrics correctly, we must set the "torchmetrics" 's metrics
        as attributes of the `module` object.
        NOTE: calling this method within this class to set "torchmetrics" 's metrics as this class's
        attributes will NOT work because of the behavior of  "PyTorch-Lightning" `LightningModule`'s `log` method.
        """
        for name, metric in metrics.items():
            if isinstance(metric, Metric | MetricCollection):
                setattr(module, name, metric)

    def load_checkpoint(self, ckpt_path: str, *args, **kwargs):
        """Implement the abstract method in `CubeTaskModule` to load a checkpoint from path."""
        return self.__class__.load_from_checkpoint(ckpt_path, **self.get_init_args())

    def get_init_args(self) -> dict:
        all_init_args: inspect.Signature = inspect.signature(self.__class__.__init__)
        filtered_init_args = {}
        for key in all_init_args.parameters:
            if key == "self":
                continue
            if not hasattr(self, key):
                raise RuntimeError("Any parameter of `__init__` must be set as attribute. ")
            filtered_init_args[key] = getattr(self, key)
        return filtered_init_args
