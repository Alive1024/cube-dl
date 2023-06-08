import torch
from torch import nn

from .wrapper_base import TaskWrapperBase

try:
    from torchmetrics.metric import Metric
    TORCHMETRICS_INSTALLED = True
except ModuleNotFoundError:
    TORCHMETRICS_INSTALLED = False


class BasicTaskWrapper(TaskWrapperBase):
    def __init__(self,
                 *,  # Compulsory keyword arguments, for better readability in config files.
                 model: nn.Module,
                 loss_function: TaskWrapperBase.LOSS_FUNCTION_T,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler=None,
                 validate_metrics: TaskWrapperBase.METRICS_T = None,
                 test_metrics: TaskWrapperBase.METRICS_T = None,
                 compile_model: bool = False):
        """
        Regular task wrapper with single optimizer (and optional single LR scheduler).

        :param model:
        :param loss_function:
        :param optimizer:
        :param lr_scheduler:
        :param validate_metrics:
        :param test_metrics:
        """
        super().__init__(model=model,
                         loss_function=loss_function,
                         validate_metrics=validate_metrics,
                         test_metrics=test_metrics,
                         compile_model=compile_model)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # In order to use the metrics in "torchmetrics" library's, the following process is must.
        if TORCHMETRICS_INSTALLED:
            self._set_torchmetrics_attrs(self, self.validate_metrics)
            self._set_torchmetrics_attrs(self, self.test_metrics)

        # `save_hyperparameters` can not be used here,
        # as the arguments are not hyper-parameters but nn.Module/Callable.
        # self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        For details on configuring LR scheduler, refer to
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html?highlight=lr_scheduler#configure-optimizers
        """
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

    def _shared_eval_step(self, batch, batch_idx, metrics: dict):
        input_data, target = batch
        predictions = self.model(input_data)
        eval_results = {}

        # After processed by superclass `TaskWrapperBase`, the passed metrics `self.validate_metrics` or
        # `self.test_metrics` must be a dict.
        for name, metric in metrics.items():
            # If the metric is from "torchmetrics", log the metric object directly and
            # let Lightning take care of when to reset the metric etc. See:
            # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#logging-torchmetrics
            if TORCHMETRICS_INSTALLED and isinstance(metric, Metric):
                metric(predictions, target)  # compute metric values
                eval_results[name] = metric  # put the object into the dict, which can be logged

            # Otherwise, log the result value.
            else:
                eval_results[name] = metric(predictions, target)

        return eval_results

    def validation_step(self, batch, batch_idx):
        eval_results = self._shared_eval_step(batch, batch_idx, metrics=self.validate_metrics)
        self.log_dict(eval_results, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        eval_results = self._shared_eval_step(batch, batch_idx, metrics=self.test_metrics)
        self.log_dict(eval_results, sync_dist=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Use data only if the batch consists of a pair of data and label
        return self.model(batch[0]) if isinstance(batch, list) and len(batch) == 2 else self.model(batch)

    @staticmethod
    def save_predictions(predictions, save_dir):
        # TODO: implement `save_predictions`
        print(type(predictions), len(predictions), predictions[0].shape)

    def predict_from_raw_data(self, src_dir, save_dir):
        # TODO: implement `predict_from_raw_data`
        raise NotImplementedError
