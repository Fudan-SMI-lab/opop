from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import JsonValue


@runtime_checkable
class TensorValue(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...
    def numel(self) -> int: ...
    def element_size(self) -> int: ...
    def detach(self) -> "TensorValue": ...
    def clone(self) -> "TensorValue": ...
    def cpu(self) -> "TensorValue": ...


@runtime_checkable
class CacheValue(Protocol):
    def get_seq_length(self, layer_idx: int = 0) -> int: ...


type Value = TensorValue | CacheValue | str | int | float | bool | None | tuple[Value, ...] | list[Value] | dict[str, Value]
type Forward = Callable[..., Value]


class ForwardModule(Protocol):
    forward: Forward
    def named_parameters(self) -> Iterable[tuple[str, Value]]: ...


class NamedModel(Protocol):
    def named_modules(self) -> Iterable[tuple[str, ForwardModule]]: ...


@dataclass(frozen=True, slots=True)
class OperatorCall:
    args: tuple[Value, ...]
    kwargs: Mapping[str, Value]


@dataclass(frozen=True, slots=True)
class OperatorSite:
    module: ForwardModule
    weights: Mapping[str, Value]
    module_path: str


@runtime_checkable
class Replacement(Protocol):
    def __call__(self, site: OperatorSite, call: OperatorCall, params: Mapping[str, JsonValue]) -> Value: ...


class FixtureCodec(Protocol):
    def size(self, call: OperatorCall) -> int: ...
    def clone(self, call: OperatorCall) -> OperatorCall: ...
    def state(self, call: OperatorCall) -> Value: ...
    def check(self, expected: Value, actual: Value) -> None: ...
    def check_state(self, expected: Value, actual: Value) -> None: ...
    def describe(self, call: OperatorCall) -> str: ...


@dataclass(frozen=True, slots=True)
class OperatorFixture:
    module_path: str
    call: OperatorCall
    expected: Value
    touched_state: Value
