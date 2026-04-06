from functools import wraps
from time import perf_counter
from typing import TypeVar
import torch

grey = "\x1b[38;20m"
green = "\x1b[32;1m" 
gray = "\x1b[38;5;240m"
blue = "\x1b[34;20m"
yellow = "\x1b[33;20m"
red = "\x1b[31;20m"
bold_red = "\x1b[31;1m"
reset = "\x1b[0m"

F = TypeVar('F')
def decorator(cls, func: F) -> F:
    @wraps(func)
    def decorated_func(*args, **kwargs):
        with cls:
            return func(*args, **kwargs)
    return decorated_func

class CPUTimer:
    def __init__(self, prefix='', synchronize=False):
        self.prefix = prefix
        self.synchronize = synchronize

    def __enter__(self):
        if self.synchronize:
            torch.cuda.synchronize()
        self.t = perf_counter()

    def __exit__(self, _type, _value, _traceback):
        if self.synchronize:
            torch.cuda.synchronize()
        t = perf_counter() - self.t
        print(f'{gray}>>> {self.prefix} {t:.6f} >>>{reset}')

    def __call__(self, func: F) -> F:
        return decorator(self, func)