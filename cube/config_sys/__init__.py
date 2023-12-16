"""
This package implements the config system.
"""
from .config_decorators import (
    data_wrapper_getter,
    model_getter,
    root_config_getter,
    task_wrapper_getter,
    trainer_getter,
)
from .root_config import RootConfig
from .shared_config import shared_config
from .utils import LazyInstance, iterable_to_generator

__all__ = [
    "RootConfig",
    "model_getter",
    "task_wrapper_getter",
    "data_wrapper_getter",
    "trainer_getter",
    "root_config_getter",
    "iterable_to_generator",
    "LazyInstance",
    "shared_config",
]
