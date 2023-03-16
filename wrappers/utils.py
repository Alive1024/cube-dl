import inspect
from functools import wraps
from pprint import pprint
import sys


def wrapper(func):
    @wraps(func)
    def collect_fn(*args, **kwargs):
        # Capture the actual arguments passed in
        # print(inspect.getargvalues(inspect.currentframe()))
        # locals() 等价于 inspect.getargvalues(inspect.currentframe()).locals
        try:
            raise TypeError
        except TypeError:
            tb = sys.exc_info()[2]

            # tb = inspect.getframeinfo(inspect.currentframe())
            # print(type(tb))
            print(inspect.getinnerframes(tb))
        # print(inspect.stack())x
        # print(inspect.getargvalues(inspect.currentframe()))
        # inspect.signature(func)  default

            return func(*args, **kwargs)
    return collect_fn
