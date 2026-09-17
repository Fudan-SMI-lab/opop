"""Process-scoped admission at existing external calls; admitted work drains unchanged."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import Final

from kernel_optimizer.agents.runtime import OpencodeClient
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from scripts.experiments.c2_method_protocol import Deadline

_CELL_LOCK: Final = Lock()


def admitted[**P, R](call: Callable[P, R], deadline: Deadline) -> Callable[P, R]:
    @wraps(call)
    def invoke(*args: P.args, **kwargs: P.kwargs) -> R:
        deadline.check()
        return call(*args, **kwargs)
    return invoke


@contextmanager
def admission(deadline: Deadline) -> Iterator[None]:
    """Cover internally constructed workers without copying acquisition or native loops.

    One cell owns this process until all generation threads drain. This guards new
    host submissions, including retries; already admitted worker waits are drain,
    not a promise to terminate their GPU process at the scheduling deadline.
    """
    with _CELL_LOCK:
        original_prompt = OpencodeClient.prompt
        original_job = WslGpuWorker.run_job
        OpencodeClient.prompt = admitted(original_prompt, deadline)
        WslGpuWorker.run_job = admitted(original_job, deadline)
        try:
            yield
        finally:
            OpencodeClient.prompt = original_prompt
            WslGpuWorker.run_job = original_job
