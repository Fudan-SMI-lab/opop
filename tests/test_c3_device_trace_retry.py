"""`prove` retries CUDA collection rather than rejecting a kernel on one empty trace.

WHY THIS EXISTS
---------------
`require_device_proof` rejects a declaration whose `kernel_name` is absent from `cuda_names`. On the
real box that tuple has been observed EMPTY while the CPU side of the same trace was complete --
`cuLaunchKernelEx` present, `Activity Buffer Request` present, 11-12 CPU events -- so the kernel
launched and the profiler ran, yet no device activity record arrived in time to be returned. Seven
probes ruled out every reconstructable cause (a profiling server, the candidate's bytes, the helper
RPC, a fresh profiler per session, reading inside vs outside the context, and each layer of the
context stack), which leaves an asynchronous flush that sometimes misses its window.

These tests pin the resulting behaviour WITHOUT a GPU: `torch` and `torch.profiler` are replaced at
the import boundary `TorchLocalDevice` uses, so the retry loop, the per-attempt record and the
preserved failure mode are all exercised as production code.

What is deliberately NOT relaxed: when no attempt sees the kernel, `cuda_names` stays empty and the
gate still rejects. Retrying buys more chances to observe activity that already happened; it never
lowers the bar for what counts as observed.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.c3_qwen3.device_gate import launch_route, require_device_proof
from examples.c3_qwen3.runner_records import RunnerError


class Event:
    def __init__(self, name: str, device_type: str) -> None:
        self.name, self.device_type = name, device_type


class Trace:
    """A `torch.profiler.profile` stand-in whose event list is scripted per session."""

    def __init__(self, sessions: list[list[Event]], log: list[int]) -> None:
        self.sessions, self.log = sessions, log
        self.mine: list[Event] = []

    def __enter__(self):
        self.mine = self.sessions.pop(0) if self.sessions else []
        self.log.append(len(self.mine))
        return self

    def __exit__(self, *exc) -> None:
        return None

    def events(self):
        return list(self.mine)


class Device:
    """The `torch.cuda.device` / `inference_mode` context managers, reduced to no-ops."""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class CudaName(str):
    """A `torch.device` stand-in: a string that also carries `.type` and `.index`.

    `prove` compares `str(self.device)` against the resident runner's device, and `ready()` reads
    `.type`/`.index`, so the stub has to satisfy both without a real torch.
    """

    type = "cuda"
    index = 0


def make_device(sessions: list[list[Event]], attempts: int):
    """A real `TorchLocalDevice` whose torch/profiler imports are the stubs above."""
    module = importlib.import_module("examples.c3_qwen3.local_device")
    entered: list[int] = []
    profiler = SimpleNamespace(
        profile=lambda **kwargs: Trace(sessions, entered),
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"))
    torch = SimpleNamespace(
        device=CudaName,
        cuda=SimpleNamespace(device=lambda d: Device(), synchronize=lambda d: None, is_initialized=lambda: True),
        inference_mode=lambda: Device())
    device = module.TorchLocalDevice.__new__(module.TorchLocalDevice)
    device.torch, device.profiler = torch, profiler
    device.device = CudaName("cuda:0")
    device.trace_attempts = attempts
    device.counters = {"jit_calls": 0, "compile_calls": 0, "launch_calls": 0, "compile_wall_us": 0}
    return module, device, entered


class Tools:
    def clone(self, call):
        return call

    def check(self, expected, actual) -> None:
        assert expected == actual

    def check_state(self, expected, actual) -> None:
        assert expected == actual

    def state(self, call):
        return ()


class Observer:
    """A `TritonObserver` stand-in: it records the per-attempt reset and the compiled kernel name."""

    def __init__(self, name: str = "declared_kernel") -> None:
        self.compiled = SimpleNamespace(name=name, n_regs=32, metadata=SimpleNamespace(shared=1024))
        self.counters = {"jit_calls": 1, "compile_calls": 1, "launch_calls": 1, "compile_wall_us": 5}
        self.stored_outputs, self.read_inputs = {1}, {0}
        self.skip = False
        self.resets, self.inputs = 0, 0

    def observe(self):
        return Device()

    def begin_attempt(self) -> None:
        self.resets += 1

    def set_inputs(self, call) -> None:
        self.inputs += 1

    def output_matches(self, output) -> bool:
        return True


def invocation_for(calls: list[str], observer: Observer, *, skip_control_detects: bool = True):
    fixture = SimpleNamespace(description=SimpleNamespace(tensors=()),
        original=SimpleNamespace(call="call", expected="call", touched_state=(), module_path="scale"))

    def candidate(call):
        calls.append(call)
        # When `observer.skip` is set, `prove` is running the skip control: the kernel's stores are
        # suppressed, so a candidate that really depends on its kernel must now return the WRONG
        # answer. A candidate that returns the right answer anyway did NOT get its output from the
        # kernel -- that is the dummy/fallback case, and `skip_control_detects=False` models it.
        if observer.skip and skip_control_detects:
            return "suppressed"
        return "call"

    return SimpleNamespace(fixture=fixture, candidate=candidate)

    return SimpleNamespace(fixture=fixture, candidate=candidate)


def runner_for(observer: Observer):
    binding = SimpleNamespace(diagnostic=lambda: Device(), counts={"scale": {"replacement_calls": 1, "old_calls": 0}})
    return SimpleNamespace(backend=SimpleNamespace(device="cuda:0"), binding=binding)


def declaration():
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
    return DeviceKernelDeclaration(backend="triton", source_file="operators.py", entry="kernel",
                                   output_arg_indices=(1,))


def run(sessions: list[list[Event]], *, attempts: int = 3, name: str = "declared_kernel",
        skip_control_detects: bool = True):
    module, device, entered = make_device(sessions, attempts)
    device.tools = Tools()
    observer = Observer(name)
    module_ref = module
    original = module_ref.TritonObserver
    module_ref.TritonObserver = lambda runner, decl, inv: observer
    try:
        calls: list[str] = []
        evidence = device.prove(runner_for(observer), declaration(),
                                invocation_for(calls, observer, skip_control_detects=skip_control_detects))
    finally:
        module_ref.TritonObserver = original
    return evidence, observer, entered


KERNEL = [Event("aten::view", "DeviceType.CPU"), Event("cuLaunchKernelEx", "DeviceType.CPU"),
          Event("declared_kernel", "DeviceType.CUDA")]
CPU_ONLY = [Event("aten::view", "DeviceType.CPU"), Event("cuLaunchKernelEx", "DeviceType.CPU"),
            Event("cudaStreamSynchronize", "DeviceType.CPU")]


def test_an_empty_cuda_trace_is_retried_and_the_recovered_attempt_passes_the_gate() -> None:
    # Given: the first two profiled executions return the CPU side only -- the exact live shape,
    # launch call present and no device record -- and the third collects the kernel.
    evidence, observer, entered = run([list(CPU_ONLY), list(CPU_ONLY), list(KERNEL)])
    # Then: the proof reports the attempt that observed the kernel, and the gate accepts it.
    assert evidence.cuda_names == ("declared_kernel",)
    assert (evidence.trace_events, evidence.trace_cuda_events) == (3, 1)
    require_device_proof(evidence)
    # ...and every attempt is on the record, so a reader can see collection was flaky rather than
    # taking the successful read on trust.
    assert [a.index for a in evidence.trace_attempts] == [0, 1, 2]
    assert [a.cuda_events for a in evidence.trace_attempts] == [0, 0, 1]
    assert [a.kernel_seen for a in evidence.trace_attempts] == [False, False, True]
    assert all(a.events == 3 for a in evidence.trace_attempts)
    # Each attempt re-seeded the observer, so the judged attempt's tensors are the ones recorded.
    # `set_inputs` runs once more than that: the skip control is a fourth execution with its own
    # fresh call, outside the profiled loop.
    assert observer.resets == 3 and observer.inputs == 4 and entered == [3, 3, 3]


def test_a_kernel_observed_on_the_first_attempt_is_not_re_executed() -> None:
    # Given: collection works immediately.
    evidence, observer, entered = run([list(KERNEL), list(KERNEL), list(KERNEL)])
    # Then: no further attempt is made -- a second read could only re-confirm the first.
    assert evidence.cuda_names == ("declared_kernel",) and len(evidence.trace_attempts) == 1
    assert observer.resets == 1 and entered == [3]
    require_device_proof(evidence)


def test_a_kernel_that_never_appears_in_the_trace_leaves_cuda_names_empty() -> None:
    # Given: the declared kernel is absent from all three traces, which collected CPU events only.
    evidence, observer, _ = run([list(CPU_ONLY), list(CPU_ONLY), list(CPU_ONLY)])
    # Then: retrying does not invent a name. `cuda_names` stays empty and every attempt is recorded
    # as having collected nothing, which is what tells a reader the INSTRUMENT was silent here rather
    # than that the kernel was missing from a working trace.
    assert evidence.cuda_names == () and evidence.trace_cuda_events == 0
    assert len(evidence.trace_attempts) == 3 and not any(a.kernel_seen for a in evidence.trace_attempts)
    assert all(a.cuda_events == 0 for a in evidence.trace_attempts)
    # The gate then falls back to the causal chain (counted launch + failing skip control), which the
    # stub observer satisfies. `tests/test_c3_device_gate_two_routes.py` owns that decision and its
    # asymmetry; here the point is only that this proof carries the fields the fallback reads.
    assert evidence.launches >= 1 and evidence.skip_control_rejected and evidence.compiled
    assert launch_route(evidence) == "causal_chain"
    require_device_proof(evidence)


def test_a_silent_trace_still_rejects_a_kernel_whose_output_is_unused() -> None:
    # Given: a silent profiler AND a skip control that PASSED -- suppressing the kernel's stores left
    # the output correct, so the output did not come from it. Neither route may admit that.
    evidence, _, _ = run([list(CPU_ONLY), list(CPU_ONLY), list(CPU_ONLY)], skip_control_detects=False)
    assert evidence.cuda_names == () and evidence.skip_control_rejected is False
    assert launch_route(evidence) is None
    with pytest.raises(RunnerError):
        require_device_proof(evidence)


def test_cuda_activity_under_a_different_name_is_reported_not_silently_accepted() -> None:
    # Given: the trace carries a CUDA record, but for some other kernel.
    other = [Event("cuLaunchKernelEx", "DeviceType.CPU"), Event("someone_elses_kernel", "DeviceType.CUDA")]
    evidence, _, _ = run([list(other), list(other), list(other)])
    # Then: collection plainly worked (cuda_events > 0), which is what distinguishes "the profiler
    # saw nothing" from "the profiler saw work that is not the declared kernel" -- opposite fixes.
    assert evidence.trace_cuda_events == 1 and evidence.cuda_names == ("someone_elses_kernel",)
    assert all(a.cuda_events == 1 and not a.kernel_seen for a in evidence.trace_attempts)
    with pytest.raises(RunnerError, match="CUDA-trace-confirmed"):
        require_device_proof(evidence)


def test_a_single_attempt_still_executes_exactly_once() -> None:
    # Given: `trace_attempts=1`, the original one-shot behaviour.
    evidence, observer, entered = run([list(CPU_ONLY), list(KERNEL)], attempts=1)
    # Then: exactly one profiled execution, and the second scripted session is never reached -- so the
    # retry is a default that a caller can turn off, not something the code does unconditionally.
    assert len(evidence.trace_attempts) == 1 and observer.resets == 1 and entered == [3]
    assert evidence.cuda_names == () and evidence.trace_attempts[0].cuda_events == 0


def test_attempts_below_one_are_refused() -> None:
    module = importlib.import_module("examples.c3_qwen3.local_device")
    device = module.TorchLocalDevice.__new__(module.TorchLocalDevice)
    # A zero-attempt device would return an unjudged proof, so it is rejected at construction rather
    # than producing evidence nobody looked at. `torch` is absent on this box, so the guard is read
    # from the constructor's own source: the point is that the check exists ahead of any device work.
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert 'raise RunnerError("device proof needs at least one profiled attempt")' in source
    assert "if trace_attempts < 1:" in source
    assert device.__class__.__init__.__kwdefaults__ == {"trace_attempts": 3}
