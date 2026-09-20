import hashlib
import importlib
import json
from collections.abc import Iterator
from typing import assert_never

from .device_records import TensorDescription
from .operator_types import CacheValue, OperatorCall, TensorValue, Value
from .runner_records import RunnerError
from .torch_fixtures import TorchFixtureCodec


def tensor_leaves(value: Value, path: str = "") -> Iterator[tuple[str, TensorValue]]:
    match value:
        case CacheValue():
            raise RunnerError("unsupported fixture: KV/cache objects are not pure-tensor calls")
        case TensorValue():
            yield path, value
        case list() | tuple():
            for i, item in enumerate(value):
                yield from tensor_leaves(item, f"{path}/{i}")
        case dict():
            for key, item in value.items():
                yield from tensor_leaves(item, f"{path}/{key}")
        case str() | bool() | int() | float() | None:
            return
        case unreachable:
            assert_never(unreachable)


class TorchPureTools(TorchFixtureCodec):
    def size(self, call: OperatorCall) -> int:
        storage = {}
        for _, tensor in tensor_leaves((call.args, dict(call.kwargs))):
            backing = getattr(tensor, "untyped_storage")()
            storage[(str(getattr(tensor, "device")), backing.data_ptr())] = backing.nbytes()
        return sum(storage.values())

    def require_pure(self, call: OperatorCall) -> None:
        leaves = tuple(tensor_leaves((call.args, dict(call.kwargs))))
        if not leaves:
            raise RunnerError("unsupported fixture: no tensor input")
        if self.size(call) > 64 * 1024 * 1024:
            raise RunnerError("unsupported fixture: 64 MiB call cap exceeded")

    def describe_tensors(self, call: OperatorCall) -> tuple[TensorDescription, ...]:
        return tuple(TensorDescription(path=path, shape=tuple(t.shape), stride=tuple(getattr(t, "stride")()),
                                       dtype=str(getattr(t, "dtype")), device=str(getattr(t, "device")))
                     for path, t in tensor_leaves((call.args, dict(call.kwargs))))

    def digest(self, call: OperatorCall) -> str:
        self.require_pure(call)
        digest = hashlib.sha256()
        for description, (_, tensor) in zip(self.describe_tensors(call), tensor_leaves((call.args, dict(call.kwargs))), strict=True):
            digest.update(description.model_dump_json().encode())
            raw = getattr(tensor.detach().cpu(), "contiguous")().view(self.torch.uint8).numpy().tobytes()
            digest.update(raw)
        scalars = []
        def visit(value: Value) -> None:
            match value:
                case TensorValue():
                    scalars.append("tensor")
                case tuple() | list():
                    for item in value:
                        visit(item)
                case dict():
                    for key, item in value.items():
                        scalars.append(key)
                        visit(item)
                case CacheValue():
                    raise RunnerError("cache state is unsupported")
                case str() | bool() | int() | float() | None:
                    scalars.append(value)
                case unreachable:
                    assert_never(unreachable)
        visit((call.args, dict(call.kwargs)))
        digest.update(json.dumps(scalars, allow_nan=False).encode())
        return digest.hexdigest()
