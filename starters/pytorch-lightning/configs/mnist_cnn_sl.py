import os.path as osp

import pytorch_lightning as pl
from cube_dl.config_sys import (
    RootConfig,
    cube_model,
    cube_root_config,
    cube_runner,
    cube_task_module,
    shared_config,
)
from cube_dl.core import CUBE_CONTEXT
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch import nn, optim
from torchmetrics import ClasswiseWrapper, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.datasets.mnist import MNIST

from callbacks.metrics_csv import MetricsCSVCallback
from models.cnn_example import ExampleCNN
from tasks.supervised_learning import SupervisedLearningTask
from utils.logger import get_csv_logger

from .components.mnist_data_module import get_mnist_data_module


@cube_model
def get_model() -> nn.Module:
    return ExampleCNN(num_input_channels=1, num_classes=shared_config.get("num_classes"))


@cube_task_module
def get_task_module():
    model = get_model()
    optimizer = optim.SGD(
        model.parameters(),
        lr=shared_config.get("lr_bs32") / 32 * shared_config.get("batch_size"),
    )
    num_classes = shared_config.get("num_classes")
    return SupervisedLearningTask(
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=shared_config.get("max_epochs"), eta_min=1e-7
        ),
        validate_metrics={
            "val_loss": nn.CrossEntropyLoss(),
            "val_acc": MetricCollection(
                {
                    "mean_acc": MulticlassAccuracy(num_classes=num_classes),
                    "classwise_acc": ClasswiseWrapper(
                        MulticlassAccuracy(num_classes=num_classes, average=None),
                        labels=MNIST.classes,
                        prefix="cw_acc_",
                    ),
                },
                prefix="val_",
            ),
        },
        test_metrics=MetricCollection(
            {
                "mean_acc": MulticlassAccuracy(num_classes=num_classes),
                "classwise_acc": ClasswiseWrapper(
                    MulticlassAccuracy(num_classes=num_classes, average=None), labels=MNIST.classes, prefix="cw_acc_"
                ),
                "f1": MulticlassF1Score(num_classes=num_classes),
            },
            prefix="test_",
        ),
    )


@cube_runner
def get_fit_runner():
    run = CUBE_CONTEXT["run"]
    return pl.Trainer(
        accelerator="auto",
        max_epochs=shared_config.get("max_epochs"),
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                dirpath=osp.join(run.run_dir, "checkpoints"),
                filename="{epoch}-{step}-{val_mean_acc:.4f}",
                save_top_k=1,
                monitor="val_mean_acc",
                mode="max",
            ),
        ],
        logger=get_csv_logger(run),
    )


@cube_runner
def get_test_runner():
    """
    "It is recommended to validate on single device to ensure each sample/batch gets evaluated exactly once."
    See: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-loop
    """
    run = CUBE_CONTEXT["run"]
    return pl.Trainer(
        accelerator="auto",
        devices=1,
        num_nodes=1,
        callbacks=[RichProgressBar()],
        logger=get_csv_logger(run),
    )


@cube_root_config
def get_root_config():
    shared_config.set("max_epochs", 25)
    shared_config.set("num_classes", 10)
    shared_config.set("batch_size", 32)
    shared_config.set("lr_bs32", 1e-3)
    return RootConfig(
        model_getters=get_model,
        task_module_getter=get_task_module,
        data_module_getter=get_mnist_data_module,
        fit_runner_getter=get_fit_runner,
        test_runner_getter=get_test_runner,
        seed_func=pl.seed_everything,
        global_seed=42,
        callbacks=MetricsCSVCallback(),
    )
