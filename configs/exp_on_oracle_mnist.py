from config_sys.root_config import RootConfig
from config_sys import root_config_getter, shared_config
from .components.task_wrappers.basic_task_wrapper import get_task_wrapper_instance
from .components.data_wrappers.oracle_mnist import get_data_wrapper_instance
from .components.trainers.basic_trainer import get_trainer_instance


@root_config_getter
def get_root_config_instance():
    # Set all the values needed to be shared among config components.
    shared_config.set("num_classes", 10)
    shared_config.set("fit_max_epochs", 5)
    return RootConfig(
        task_wrapper_getter=get_task_wrapper_instance,
        data_wrapper_getter=get_data_wrapper_instance,
        default_trainer_getter=get_trainer_instance,
    )
