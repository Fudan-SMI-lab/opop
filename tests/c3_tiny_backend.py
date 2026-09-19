from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from math import isclose

from examples.c3_qwen3.runner_records import RunnerError


class Scale:
    def __init__(self) -> None:
        self.forward = self.original
        self.weight = 2.0

    def original(self, value: float) -> float:
        return value * self.weight

    def named_parameters(self) -> Iterator[tuple[str, float]]:
        yield "weight", self.weight


class TinyModel:
    def __init__(self) -> None:
        self.scale = Scale()

    def named_modules(self) -> Iterator[tuple[str, Scale]]:
        yield "scale", self.scale


class NumericCodec:
    def size(self, call) -> int:
        return 0

    def clone(self, call):
        return deepcopy(call)

    def state(self, call):
        return (call.args, dict(call.kwargs))

    def check(self, expected, actual) -> None:
        if not isclose(expected, actual, rel_tol=0.02, abs_tol=0.02):
            raise RunnerError(f"local numerical mismatch: {expected} != {actual}")

    def check_state(self, expected, actual) -> None:
        if expected != actual:
            raise RunnerError("local state mismatch")

    def describe(self, call) -> str:
        return "scalar"


class TinyBackend:
    """Numerical autoregressive model: cumulative token history determines next logits."""

    def __init__(self) -> None:
        self.model = TinyModel()
        self.codec = NumericCodec()
        self.rows: list[list[int]] = []
        self.position_history: list[tuple[int, ...]] = []
        self.batch_history: list[int] = []
        self.loads = 1
        self.forward_calls = 0
        self.closed = False
        self.now = 0.0

    def inference(self):
        return nullcontext()

    def clock(self) -> float:
        return self.now

    def synchronize(self) -> None:
        self.now += 0.001

    def reset(self) -> None:
        self.rows = []

    def begin(self, rows: Sequence[Sequence[int]]) -> None:
        self.rows = [[] for _ in rows]
        self.batch_history.append(len(rows))

    def forward(self, rows: Sequence[Sequence[int]]) -> tuple[tuple[float, ...], ...]:
        self.forward_calls += 1
        if len(rows) != len(self.rows):
            raise RunnerError("batch state mismatch")
        for history, incoming in zip(self.rows, rows, strict=True):
            history.extend(incoming)
        self.position_history.append(tuple(map(len, self.rows)))
        self.now += 0.01 if len(rows[0]) > 1 else 0.002
        return tuple(tuple(-abs(v - int(self.model.scale.forward(float(sum(row)))) % 7)
                           for v in range(7)) for row in self.rows)

    def greedy(self, logits: Sequence[Sequence[float]]) -> tuple[int, ...]:
        return tuple(max(range(len(row)), key=lambda i: row[i]) for row in logits)

    def cpu_logits(self, logits: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(row) for row in logits)

    def cache_length(self) -> int:
        return len(self.rows[0])

    def weights_stamp(self) -> tuple[float, ...]:
        return (self.model.scale.weight,)

    def close(self) -> None:
        self.closed = True
        self.reset()
