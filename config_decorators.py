"""
Decorators for Getters in Config Files.
"""

from typing import Callable
import inspect

from root_config import RootConfig


def root_config_getter(getter_func: Callable):
    # Check the name of the function
    if getter_func.__name__ != RootConfig.ROOT_CONFIG_GETTER_NAME:
        raise ValueError(f"The root config getter's must be named `{RootConfig.ROOT_CONFIG_GETTER_NAME}`, "
                         f"e.g. `def {RootConfig.ROOT_CONFIG_GETTER_NAME}(): ...`")
    return getter_func


def model_getter(getter_func: Callable):
    """
    This decorator does nothing at present, defined just for symmetry and future extension.
    """
    return getter_func


def task_wrapper_getter(model_getter_func: Callable):
    def decorated_fn(getter_func: Callable):
        # Set the "model_getter" attribute to the function
        getter_func.model_getter_func = model_getter_func
        return getter_func

    return decorated_fn


def data_wrapper_getter(getter_func: Callable):
    """
    This decorator does nothing at present, defined just for symmetry and future extension.
    """
    return getter_func


def trainer_getter(getter_func: Callable):
    # Ensure the trainer getter has a parameter named "logger" to allow passing logger(s)
    param_names = inspect.signature(getter_func).parameters.keys()
    if (len(param_names) == 0) or ("logger" not in param_names):
        raise RuntimeError("The trainer getter must have a parameter named `logger`, e.g. "
                           "`@trainer_getter\ndef get_trainer_instance(logger):\n\t...`")
    return getter_func
