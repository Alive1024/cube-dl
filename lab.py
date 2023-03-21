import inspect
import sys
from collections import OrderedDict
from typing import Iterable, Callable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar


def inspect_func(func):
    all_init_args: inspect.Signature = inspect.signature(func)
    print(all_init_args.parameters.keys())


def _collect_fn(key, value, dst_dict: OrderedDict):
    #
    if isinstance(value, (bool, int, float, complex, str)):
        dst_dict[key] = value

    #
    elif isinstance(value, dict):
        dst_dict[key] = OrderedDict()
        for k, v in value.items():
            # dst_dict[key][k] = OrderedDict()
            _collect_fn(k, v, dst_dict[key])
    elif isinstance(value, Iterable):  # list, tuple, set
        dst_dict[key] = OrderedDict()
        for idx, ele in enumerate(value):
            # dst_dict[key][idx] = OrderedDict()
            _collect_fn(idx, ele, dst_dict[key])

    #
    # elif isinstance(value, Callable):
    #     print(4)
    #     dst_dict[key] = OrderedDict()
    #     for k, v in inspect.signature(value).parameters.items():
    #         # dst_dict[key][k] = OrderedDict()
    #         _collect_fn(k, v, dst_dict[key])

    # object
    else:
        dst_dict[key] = OrderedDict()
        for k, v in inspect.signature(value.__class__.__init__).parameters.items():
            print(k)
            if k not in ("self", "args", "kwargs"):
                # dst_dict[key][k] = OrderedDict()
                try:
                    attr_value = getattr(value, k)
                    _collect_fn(k, attr_value, dst_dict[key])
                except AttributeError:
                    # print(v.default)
                    pass


class Alice:
    def __init__(self, a, b=42):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Bob:
    def __init__(self, c, alice):
        pass


def get_trainer_instance():
    return pl.Trainer(
        callbacks=[RichProgressBar(leave=True)]
    )


def _collect_trainer_args(frame, event, arg):
    """
    Similar to `_collect_task_wrapper_args`.
    """
    if event != "return":
        return
    if frame.f_back.f_code.co_name not in ("insert_env_defaults", "get_trainer_instance"):
        return

    # if frame.f_code.co_name != "__init__":
    #     return

    f_locals = frame.f_locals
    print(frame.f_code.co_name)
    print(f_locals)
    # if "self" in f_locals:
    #     self._init_local_vars[id(f_locals["self"])] = f_locals
    #     if type(f_locals["self"]) == pl.Trainer:
    #         self._parse_frame_locals(f_locals, part="trainer")


def func() -> dict:
    pass


if __name__ == '__main__':
    # from configs.exp_on_oracle_mnist import get_config_instance
    # from entities import Run
    #
    # config_instance = get_config_instance()
    # config_instance.setup_wrappers()
    # config_instance.setup_trainer("csv", Run(name="dev", desc="123", proj_id=1, exp_id=1,
    #                                          job_type="fit",
    #                                          output_dir="/Users/yihaozuo/Zyh-Coding-Projects/Personal/DL-Template"
    #                                                     "-Project/outputs"))

    # for k, v in inspect.signature(pl.Trainer.__init__).parameters.items():
    #     print(k, v)

    sys.setprofile(_collect_trainer_args)
    trainer = get_trainer_instance()
    sys.setprofile(None)
