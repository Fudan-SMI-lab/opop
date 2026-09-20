from collections.abc import Iterator
from contextlib import contextmanager

from .operator_types import Forward, ForwardModule


@contextmanager
def replace_forward(module: ForwardModule, forward: Forward) -> Iterator[None]:
    owned, previous = "forward" in vars(module), module.forward
    module.forward = forward
    try:
        yield
    finally:
        if owned:
            module.forward = previous
        else:
            delattr(module, "forward")
