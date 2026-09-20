import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from examples.c3_qwen3.device_records import (
    FixtureDescription, KernelEvidence, LocalRequest, LocalRuntime, PureFixture, SourceEligibility,
    StageIdentity, TensorDescription,
)
from examples.c3_qwen3.operator_types import OperatorFixture
from examples.c3_qwen3.runner_records import RunnerError
from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from tests.test_c3_model_binding import make_bundle
from tests.test_c3_model_runner import runner
from examples.c3_qwen3.pure_tensors import tensor_leaves


@dataclass(frozen=True, slots=True)
class Tensor:
    values: np.ndarray
    @property
    def shape(self):
        return self.values.shape
    @property
    def dtype(self):
        return str(self.values.dtype)
    @property
    def device(self):
        return "cpu"
    def stride(self):
        return tuple(s // self.values.itemsize for s in self.values.strides)
    def numel(self):
        return self.values.size
    def element_size(self):
        return self.values.itemsize
    def detach(self):
        return self
    def cpu(self):
        return self
    def clone(self):
        return Tensor(self.values.copy())
    def __mul__(self, value):
        return Tensor(self.values * value)


class Tools:
    def require_pure(self, call):
        if not all(isinstance(x, Tensor) for x in call.args):
            raise RunnerError("unsupported pure fixture")
    def size(self, call):
        return sum(x.values.nbytes for _, x in tensor_leaves((call.args, dict(call.kwargs))))
    def clone(self, call):
        return copy.deepcopy(call)
    def state(self, call):
        return (call.args, dict(call.kwargs))
    def check(self, expected, actual):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise RunnerError("shape/dtype mismatch")
        if not np.isfinite(actual.values).all():
            raise RunnerError("nonfinite output")
        np.testing.assert_allclose(actual.values, expected.values, rtol=.02, atol=.02)
    def check_state(self, expected, actual):
        self.check(expected[0][0], actual[0][0])
    def describe(self, call):
        return str(call.args[0].shape)
    def describe_tensors(self, call):
        return tuple(TensorDescription(path=str(i), shape=x.shape, stride=x.stride(), dtype=x.dtype, device=x.device)
                     for i, x in enumerate(call.args))
    def digest(self, call):
        return hashlib.sha256(call.args[0].values.tobytes()).hexdigest()


class Budget:
    def __init__(self, ordinal=0, allowed=True):
        self.ordinal, self.allowed, self.calls = ordinal, allowed, []
    def admit(self, stage, operation):
        self.calls.append((stage.stage, operation))
        return self.ordinal if self.allowed else None


class Device:
    def __init__(self):
        self.tools = Tools()
        self.ready_calls = 0
        self.bad_proof = {}
        self.counters = {}
    def ready(self):
        self.ready_calls += 1
    def execution(self):
        from contextlib import nullcontext
        return nullcontext()
    def identity(self):
        return {"kind": "cpu-test", "device": "simulated", "uuid": None}
    def prove(self, resident, declaration, invocation):
        invocation.candidate(self.tools.clone(invocation.fixture.original.call))
        return KernelEvidence(backend="triton", source_file=declaration.source_file, entry=declaration.entry,
            compiled=True, kernel_name="observed", launches=1, cuda_names=("observed",), output_used=True,
            stored_output_args=(1,), loaded_input_args=(0,), skip_control_rejected=True).model_copy(update=self.bad_proof)
    def time_us(self, operation):
        started = perf_counter()
        operation()
        return max((perf_counter() - started) * 1e6, .001)


def case(root: Path, body="return call.args[0] * factor"):
    from examples.c3_qwen3.operator_types import OperatorCall
    resident, backend, prepared = runner(root / "raw")
    from examples.c3_qwen3.fast_prepare import FastAccess, attach_fast, prepare_fast
    setup = Path(__file__).resolve().parents[1] / "results/c3-fast-kernel-iteration/setup"
    fast = prepare_fast(setup / "contract.json", setup / "preflight.json",
                        FastAccess(profile="c3_fast_device", phase="development"))
    attach_fast(resident, fast)
    prepared = fast.task
    backend.codec = Tools()
    resident.binding.codec = backend.codec
    resident.binding.register_site("scale", ("scale",))
    call = OperatorCall((Tensor(np.array([1., 2., 3.], dtype=np.float32)),), {})
    original = OperatorFixture("scale", call, call.args[0] * 2, backend.codec.state(call))
    description = FixtureDescription(site_id="scale", module_path="scale", goal="single", phase="decode",
        decode_step=127, batch_size=1, prompt_length=256, tensors=backend.codec.describe_tensors(call),
        source="reference multiply", source_sha256="source", contract_sha256=prepared.contract_sha256,
        corpus_sha256=prepared.corpus_sha256)
    identity = hashlib.sha256((description.model_dump_json() + backend.codec.digest(call)).encode()).hexdigest()
    fixture = PureFixture(description, original, identity, 36)
    bundle = make_bundle(root / "candidate", "def replace(site, call, params):\n    " + body + "\n")
    request = LocalRequest(mode="fixture_only", stage=StageIdentity(profile="c3_fast_device", stage="development", framework_id="F0"),
        bundle_path=bundle.path, params={}, site_id="scale", kernels=(DeviceKernelDeclaration(
            backend="triton", source_file="operators.py", entry="kernel", output_arg_indices=(1,)),),
        source_gate=SourceEligibility(eligible=True, bundle_sha256=bundle.bundle_sha256, params_sha256=bundle.params_sha256))
    return LocalRuntime(resident, Device(), Budget()), request, fixture


def tensor_backend(runtime):
    backend = runtime.runner.backend
    def forward(rows):
        backend.forward_calls += 1
        for history, row in zip(backend.rows, rows, strict=True):
            history.extend(row)
        values = np.asarray(rows, dtype=np.float32)
        tensor = Tensor(np.repeat(values[..., None], 3, axis=-1))
        result = backend.model.scale.forward(tensor)
        return tuple(tuple(-abs(i - int(row[-1, 0]) % 7) for i in range(7)) for row in result.values)
    backend.forward = forward


class Trace:
    def __init__(self):
        self.enabled = True
        self.regions = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def active(self, enabled):
        self.enabled = enabled
    def region(self, label):
        from contextlib import nullcontext
        assert self.enabled
        self.regions.append(label)
        return nullcontext()
    def export(self, path):
        path.write_text("{}")
    def records(self):
        return {"site_kernels": [], "cuda_intervals": [{"start_us": 0, "end_us": 5}, {"start_us": 2, "end_us": 8}],
                "test_backend": "CPU profiler fixture, not CUDA evidence"}
