from collections.abc import Generator, Iterable


def iterable_to_generator(seq: Iterable) -> Generator:
    """
    This function is very simple, but absolutely necessary when any iterable of
    super-large variables (e.g. tensors) appears within the getter functions.
    In these cases, it will be very slow to complete the collecting process and
    unnecessary to log these super-large variables into the "hparams.json".
    An easy way to make them avoid being collected is to convert the iterable
    to a generator, as the hyperparameter collection process will not collect
    values of generators.

    A typical case is that you assign parameter groups to an optimizer, but
    constructs a list of dicts that containing key-tensor pairs, the tensors
    may be very large as they are actual parameters in the model. Hence, in
    this case, a generator should be used instead of an iterable. You can use
    this function to convert the list to a generator.
    """
    yield from seq


class LazyInstance:
    """
    Store a class and its initialization arguments temporarily, allowing the class
    to be instantiated manually later.
    """

    def __init__(self, cls: type, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def get_cls(self):
        return self.cls

    def get_init_args(self):
        return self.args, self.kwargs

    def instantiate(self):
        return self.cls(*self.args, **self.kwargs)
