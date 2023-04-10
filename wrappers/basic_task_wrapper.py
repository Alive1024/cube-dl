from typing import Iterable, Optional, Callable, Union, Dict

import torch
from torch import nn

from .wrapper_base import TaskWrapperBase


class BasicTaskWrapper(TaskWrapperBase):
    def __init__(self,
                 *,  # Compulsory keyword arguments, for better readability in config files.
                 model: nn.Module,
                 loss_function: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler=None,
                 validate_metrics: Optional[Union[Dict[str, Callable], Iterable[Callable], Callable]] = None,
                 test_metrics: Optional[Union[Dict[str, Callable], Iterable[Callable], Callable]] = None
                 ):
        """
        Regular task wrapper with single optimizer (and optional single LR scheduler).

        :param model:
        :param loss_function:
        :param optimizer:
        :param lr_scheduler:
        :param validate_metrics:
        :param test_metrics:
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Use the loss function as the default metrics if they are not provided
        self.validate_metrics = validate_metrics if validate_metrics else [loss_function]
        self.test_metrics = test_metrics if test_metrics else [loss_function]

        # self.save_hyperparameters()   # Can't be directly used here, as the arguments

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.lr_scheduler:
            return {"optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler}
        else:
            return self.optimizer

    def training_step(self, batch, batch_idx):
        input_data, target = batch
        predictions = self.model(input_data)
        loss = self.loss_function(predictions, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx, metric_name_prefix: str, metrics):
        input_data, target = batch
        predictions = self.model(input_data)
        metric_values = {}
        if isinstance(metrics, dict):
            for name, metric in metrics.items():
                # Append the prefix to the name
                if not name.startswith(metric_name_prefix):
                    name = f"{metric_name_prefix}_{name}"
                metric_values[name] = metric(predictions, target)
        # For Iterable without name(s)
        else:
            if not isinstance(metrics, Iterable):
                metrics = [metrics]
            for metric in metrics:
                metric_values[f"{metric_name_prefix}_{metric._get_name()}"] = metric(predictions, target)

        return metric_values

    def validation_step(self, batch, batch_idx):
        metric_values = self._shared_eval_step(batch, batch_idx,
                                               metric_name_prefix="val", metrics=self.validate_metrics)
        self.log_dict(metric_values, prog_bar=True)

    def test_step(self, batch, batch_idx):
        metric_values = self._shared_eval_step(batch, batch_idx,
                                               metric_name_prefix="test", metrics=self.test_metrics)
        self.log_dict(metric_values)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Use data only if the batch consists of a pair of data and label
        return self.model(batch[0]) if isinstance(batch, list) and len(batch) == 2 else self.model(batch)

    @staticmethod
    def save_predictions(predictions, save_dir):
        # TODO: implement `save_predictions`
        print(type(predictions), len(predictions), predictions[0].shape)

    @staticmethod
    def predict_from_raw_data(src_dir, save_dir):
        # TODO: implement `predict_from_raw_data`
        raise NotImplementedError
