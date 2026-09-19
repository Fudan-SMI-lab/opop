import copy
import importlib
from collections.abc import Iterator
from typing import assert_never

from .operator_types import CacheValue, OperatorCall, TensorValue, Value
from .runner_records import RunnerError


def state_tree(value: Value) -> Value:
    match value:
        case CacheValue():
            layers = getattr(value, "layers", ())
            return {"length": value.get_seq_length(), "layers": [
                (getattr(layer, "keys", None), getattr(layer, "values", None)) for layer in layers]}
        case tuple():
            return tuple(state_tree(v) for v in value)
        case list():
            return [state_tree(v) for v in value]
        case dict():
            return {k: state_tree(v) for k, v in value.items()}
        case TensorValue() | str() | int() | float() | bool() | None:
            return value
        case unreachable:
            assert_never(unreachable)


def tensors(value: Value) -> Iterator[TensorValue]:
    match value:
        case TensorValue():
            yield value
        case tuple() | list():
            for item in value:
                yield from tensors(item)
        case dict():
            for item in value.values():
                yield from tensors(item)
        case CacheValue():
            yield from tensors(state_tree(value))
        case str() | int() | float() | bool() | None:
            return
        case unreachable:
            assert_never(unreachable)


class TorchFixtureCodec:
    def __init__(self) -> None:
        self.torch = importlib.import_module("torch")

    def state(self, call: OperatorCall) -> Value:
        return state_tree((call.args, dict(call.kwargs)))

    def size(self, call: OperatorCall) -> int:
        unique = {id(t): t for t in tensors(self.state(call))}
        return sum(t.numel() * t.element_size() for t in unique.values())

    def clone(self, call: OperatorCall) -> OperatorCall:
        if self.size(call) > 64 * 1024 * 1024:
            raise RunnerError("local fixture exceeds 64 MiB; choose a smaller explicit site/fixture")
        # One deepcopy memo preserves aliases, views and shared cache identity across args/kwargs.
        args, kwargs = copy.deepcopy((call.args, dict(call.kwargs)))
        return OperatorCall(args, kwargs)

    def check(self, expected: Value, actual: Value) -> None:
        for tensor in tensors(actual):
            if not self.torch.isfinite(tensor).all().item():
                raise RunnerError("local operator produced nonfinite output")
        self.torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02,
                                        check_dtype=True, check_device=True, equal_nan=False)

    def check_state(self, expected: Value, actual: Value) -> None:
        self.torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02,
                                        check_dtype=True, check_device=True, equal_nan=False)
        left, right = tuple(tensors(expected)), tuple(tensors(actual))
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if not self.torch.is_floating_point(a):
                self.torch.testing.assert_close(a, b, rtol=0, atol=0)
            for c, d in zip(left[:i], right[:i], strict=True):
                if ((a is c) != (b is d)
                        or (getattr(a, "untyped_storage")().data_ptr() == getattr(c, "untyped_storage")().data_ptr())
                        != (getattr(b, "untyped_storage")().data_ptr() == getattr(d, "untyped_storage")().data_ptr())):
                    raise RunnerError("local operator changed tensor alias identity")
        self._discrete_state(expected, actual)

    def _discrete_state(self, expected: Value, actual: Value) -> None:
        match expected:
            case dict():
                if not isinstance(actual, dict) or expected.keys() != actual.keys():
                    raise RunnerError("local state keys changed")
                for key, value in expected.items():
                    self._discrete_state(value, actual[key])
            case tuple() | list():
                if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
                    raise RunnerError("local state structure changed")
                for a, b in zip(expected, actual, strict=True):
                    self._discrete_state(a, b)
            case str() | int() | bool() | None:
                if expected != actual:
                    raise RunnerError("local discrete state changed")
            case TensorValue() | CacheValue() | float():
                return
            case unreachable:
                assert_never(unreachable)

    def describe(self, call: OperatorCall) -> str:
        return repr([(tuple(t.shape), str(getattr(t, "dtype", "unknown"))) for t in tensors(self.state(call))])
