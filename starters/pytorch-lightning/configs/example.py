import pytorch_lightning as pl
from cube.config_sys import (
    RootConfig,
    cube_model,
    cube_root_config,
    cube_runner,
    cube_task_module,
    shared_config,
)
from pytorch_lightning.callbacks import RichProgressBar
from torch import nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from callbacks.metrics_csv import MetricsCSVCallback
from models.cnn_example import ExampleCNN
from tasks.supervised_learning import SupervisedLearningTaskModule
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
    return SupervisedLearningTaskModule(
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=shared_config.get("max_epochs"), eta_min=1e-7
        ),
        validate_metrics=MetricCollection(
            {
                # "val_loss": nn.CrossEntropyLoss(),
                "val_acc": MulticlassAccuracy(num_classes=num_classes)
            }
        ),
        test_metrics={"test_acc": MulticlassAccuracy(num_classes)},
    )


@cube_runner
def get_fit_runner(logger):
    return pl.Trainer(
        accelerator="auto",
        max_epochs=shared_config.get("max_epochs"),
        callbacks=[RichProgressBar()],
        logger=logger,
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
        logger_getters=[get_csv_logger],
        seed_func=pl.seed_everything,
        global_seed=42,
        archive_hparams=False,
        callbacks=MetricsCSVCallback(),
    )
