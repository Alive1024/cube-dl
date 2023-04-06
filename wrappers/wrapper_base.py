import inspect

from pytorch_lightning import LightningModule


class TaskWrapperBase(LightningModule):
    def get_init_args(self) -> dict:
        all_init_args: inspect.Signature = inspect.signature(self.__class__.__init__)
        filtered_init_args = {}
        for key in all_init_args.parameters.keys():
            if key == "self":
                continue
            filtered_init_args[key] = getattr(self, key)
        return filtered_init_args
