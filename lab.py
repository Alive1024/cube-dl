import inspect
from collections import OrderedDict
from typing import Iterable, Callable
import json

from pytorch_lightning import Trainer


class Alice:
    def __init__(self, a: int, b: int = 42):
        self.a = a
        self.b = b


class Bob:
    def __init__(self, d: str, alice: Alice):
        self.d = d
        self.alice = alice


def inspect_obj(obj):
    all_init_args: inspect.Signature = inspect.signature(obj.__class__.__init__)
    # print(all_init_args.parameters.keys())

    d = OrderedDict()
    for key, value in all_init_args.parameters.items():
        if key != "self":
            d[key] = OrderedDict()
            attr_value = getattr(obj, key)

            if isinstance(attr_value, Iterable):
                for idx, ele in enumerate(attr_value):
                    d[key][idx] = {}
                    d[key][idx]["type"] = str(type(ele))
                    d[key][idx]["value"] = repr(ele)

            d[key]["type"] = str(type(attr_value))
            d[key]["value"] = repr(attr_value)

    return d


def inspect_func(func):
    all_init_args: inspect.Signature = inspect.signature(func)
    print(all_init_args.parameters.keys())


def _collect_fn(key, value, dst_dict: OrderedDict):
    #
    if isinstance(value, (bool, int, float, complex, str)):
        print(1)
        dst_dict[key] = value

    #
    elif isinstance(value, dict):
        print(2)
        dst_dict[key] = OrderedDict()
        for k, v in value.items():
            # dst_dict[key][k] = OrderedDict()
            _collect_fn(k, v, dst_dict[key])
    elif isinstance(value, Iterable):    # list, tuple, set
        print(3)
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
        print(5)
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


def collect_hparams(obj):
    hparams = OrderedDict()
    _collect_fn("hparams", obj, hparams)
    print(json.dumps(hparams, indent=4))


def getter(func: Callable):
    def target(*args, **kwargs):
        # Capture the actual arguments passed in
        print(inspect.getargvalues(inspect.currentframe()))
        return func(*args, **kwargs)
    return target


if __name__ == '__main__':
    # all_init_args: inspect.Signature = inspect.signature(Config.__init__)

    ac = Alice(1)
    bob = Bob(d="hello", alice=ac)
    #
    # all_init_args: inspect.Signature = inspect.signature(ac.__class__.__init__)
    # # print(all_init_args.parameters.keys())
    # for key, value in all_init_args.parameters.items():
    #     if key != "self":
    #         print(key, value, getattr(ac, key))

    # inspect_func(foo)
    # trainer = Trainer()
    # collect_hparams(bob)

    from configs.components.task_wrappers.basic_task_wrapper import get_task_wrapper_instance
    # from configs.components.data_wrappers.oracle_mnist import get_data_wrapper_instance
    #
    # task_wrapper = get_task_wrapper_instance()
    # # data_wrapper = get_data_wrapper_instance()
    # # collect_hparams()
    # # print(inspect_obj(task_wrapper))
    # print(json.dumps(inspect_obj(task_wrapper), indent=4))
    # # print(json.dumps(inspect_obj(data_wrapper), indent=4))
    #
    # # print(isinstance(task_wrapper, Callable))
    # collect_hparams(task_wrapper)

    @getter
    def foo(a=1, b=2):
        # cur_frame = inspect.currentframe()
        # print(inspect.getargvalues(cur_frame))
        print(a, b)

    # foo(a=42)

    get_task_wrapper_instance()
