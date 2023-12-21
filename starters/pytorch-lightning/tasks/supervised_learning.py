"""A task example based on [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)."""
from torch import nn, optim
from torchmetrics import Metric, MetricCollection

from .base import METRICS_T, TaskBase


class SupervisedLearningTaskModule(TaskBase):
    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler = None,
        validate_metrics: METRICS_T | None = None,
        test_metrics: METRICS_T | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.validate_metrics = validate_metrics
        self.test_metrics = test_metrics

        self.set_torchmetrics_attrs(self, self.validate_metrics)
        self.set_torchmetrics_attrs(self, self.test_metrics)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """For details on configuring LR scheduler, refer to
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html?highlight=lr_scheduler#configure-optimizers
        """
        if self.lr_scheduler:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
        else:
            return self.optimizer

    def training_step(self, batch, batch_idx):
        input_data, target = batch
        predictions = self.model(input_data)
        loss = self.loss_function(predictions, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx, metrics: dict | MetricCollection) -> dict:
        input_data, target = batch
        pred = self.model(input_data)

        if isinstance(metrics, MetricCollection):
            return metrics(pred, target)  # `MetricCollection` returns a flat dict

        eval_results = {}
        for name, metric in metrics.items():
            if isinstance(metric, Metric):
                metric_values = metric(pred, target)  # compute metric values

                # For non-scalar
                # Currently (2023-12), PyTorch-Lightning doesn't support log non-scalar metric value, which brings
                # inconvenience if we provide some class-wise metrics, a workaround is to "unfold" the metric result.
                if isinstance(metric_values, dict):
                    for key, val in metric_values.items():
                        eval_results[key] = val
                # For scalar
                else:
                    # Log the metric object directly and let PyTorch-Lightning take care of when to reset the metric
                    # etc. See: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#logging-torchmetrics
                    eval_results[name] = metric

            if isinstance(metric, MetricCollection):
                metric_values = metric(pred, target)  # `MetricCollection` returns a flat dict
                for key, val in metric_values.items():
                    eval_results[key] = val

            # Otherwise, log the result value.
            else:
                eval_results[name] = metric(pred, target)

        return eval_results

    def validation_step(self, batch, batch_idx):
        eval_results = self._shared_eval_step(batch, batch_idx, metrics=self.validate_metrics)
        self.log_dict(eval_results, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        eval_results = self._shared_eval_step(batch, batch_idx, metrics=self.test_metrics)
        self.log_dict(eval_results, sync_dist=True)
