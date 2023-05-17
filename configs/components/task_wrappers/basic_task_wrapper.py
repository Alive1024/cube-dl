import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from wrappers import BasicTaskWrapper
from config_sys import task_wrapper_getter
from ..models.example_cnn_oracle_mnist import get_model_instance


@task_wrapper_getter(model_getter_func=get_model_instance)
def get_task_wrapper_instance():
    model = get_model_instance()
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return BasicTaskWrapper(
        model=model,
        loss_function=loss_func,
        optimizer=optimizer,
        # lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000),
        validate_metrics=loss_func,
        # test_metrics=[loss_func]
        test_metrics=[loss_func, MulticlassAccuracy(num_classes=10)],
    )
