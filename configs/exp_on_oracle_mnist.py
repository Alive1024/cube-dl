from root_config import RootConfig

from .components.task_wrappers.basic_task_wrapper import get_task_wrapper_instance
from .components.data_wrappers.oracle_mnist import get_data_wrapper_instance
from .components.trainers.basic_trainer import get_trainer_instance


def get_root_config_instance():
    return RootConfig(
        task_wrapper_getter=get_task_wrapper_instance,
        data_wrapper_getter=get_data_wrapper_instance,
        default_trainer_getter=get_trainer_instance,
    )
