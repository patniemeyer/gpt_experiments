import random
from copy import deepcopy
import numpy as np
import torch
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

_seed_everything_value: int = 1337


def seed_everything(seed: int = _seed_everything_value):
    global _seed_everything_value
    _seed_everything_value = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_everything_value() -> int:
    return _seed_everything_value


class Config(dict):
    # Support keyword args in init
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Property style access forwarded to dict
    def __getattr__(self, k):
        return self.get(k, None)

    # Property style access forwarded to dict
    def __setattr__(self, k, v):
        self[k] = v

    # Pretty print
    def __str__(self):
        return self._str_indent()

    def _str_indent(self, indent=0):
        indent_space = ' ' * (indent * 2)
        lines = []
        for k, v in self.items():
            if isinstance(v, Config):
                lines.append(f"{indent_space}{k}:\n{v._str_indent(indent + 1)}")
            else:
                lines.append(f"{indent_space}{k}: {v}")
        return '\n'.join(lines)

    # Support overrides, e.g. default_config().add(a=1, b=2)
    def add(self, **kwargs):
        copy = deepcopy(self)
        copy.update(kwargs)
        return copy

    # Support pickle
    def __reduce__(self):
        return Config, (), None, None, iter(self.items())


if __name__ == '__main__':
    c = Config(a=1, b=2)
    c.foo = 42
    c.bar = 43
    c.model = Config()
    c.model.batch_size = 100
    print(c)
    a = Config(a=99)
    c += a
    print(c)


def printe(*args, **kwargs):
    print(*args, **kwargs)
    exit()
