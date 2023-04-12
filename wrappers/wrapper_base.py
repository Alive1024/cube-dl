import inspect
from abc import ABCMeta, abstractmethod

from pytorch_lightning import LightningModule


class TaskWrapperBase(LightningModule, metaclass=ABCMeta):
    def get_init_args(self) -> dict:
        all_init_args: inspect.Signature = inspect.signature(self.__class__.__init__)
        filtered_init_args = {}
        for key in all_init_args.parameters.keys():
            if key == "self":
                continue
            filtered_init_args[key] = getattr(self, key)
        return filtered_init_args

    @staticmethod
    @abstractmethod
    def save_predictions(predictions, save_dir):
        pass

    @abstractmethod
    def predict_from_raw_data(self, src_dir, save_dir):
        pass
