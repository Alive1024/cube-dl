"""Decorators for getters in config files."""

import inspect
from collections.abc import Callable

from .root_config import RootConfig


def cube_root_config(getter_func: Callable):
    # Check the name of the function
    if getter_func.__name__ != RootConfig.ROOT_CONFIG_GETTER_NAME:
        raise ValueError(
            f"The root config getter's must be named `{RootConfig.ROOT_CONFIG_GETTER_NAME}`, "
            f"e.g. `def {RootConfig.ROOT_CONFIG_GETTER_NAME}(): ...`"
        )
    return getter_func


def cube_model(getter_func: Callable):
    """Do nothing at present, defined for symmetry and future extension."""
    return getter_func


def cube_task_module(getter_func: Callable):
    param_names = inspect.signature(getter_func).parameters.keys()
    if "model" not in param_names:
        raise RuntimeError(
            "The task module must have a parameter named `model`, e.g.\n"
            "@cube_task_module\ndef get_task_module(model):\n\t..."
        )
    return getter_func


def cube_data_module(getter_func: Callable):
    """Do nothing at present, defined for symmetry and future extension."""
    return getter_func


def cube_runner(getter_func: Callable):
    """Do nothing at present, defined for symmetry and future extension."""
    return getter_func
