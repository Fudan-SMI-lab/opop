import importlib
import importlib.metadata
import socket
from collections.abc import Callable
from contextlib import contextmanager

from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from .device_records import KernelEvidence, LocalInvocation, TraceAttempt
from .model_runner import ResidentRunner
from .operator_types import Value
from .pure_tensors import TorchPureTools
from .runner_records import RunnerError
from .triton_observer import TritonObserver

# The CPU-side profiler events that mean "a kernel launch was issued to the driver". These come from
# the profiler itself, not from our patched `JITFunction.run`, so they are independent corroboration
# that a launch happened even when the device-activity side of the same trace comes back empty --
# which is exactly the observed failure (14 of 15 live proves on box4 carried `cuLaunchKernelEx` with
# no CUDA activity records at all). `device_gate.launch_route` requires one of these before it will
# accept the causal-chain route.
LAUNCH_API_EVENTS = frozenset({
    "cuLaunchKernelEx", "cuLaunchKernel", "cuLaunchCooperativeKernel",
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
})


class TorchLocalDevice:
    def __init__(self, device: str, *, trace_attempts: int = 3) -> None:
        self.torch = importlib.import_module("torch")
        self.profiler = importlib.import_module("torch.profiler")
        self.device = self.torch.device(device)
        self.tools = TorchPureTools()
        self.counters = {"jit_calls": 0, "compile_calls": 0, "launch_calls": 0, "compile_wall_us": 0}
        # How many times `prove` will execute-and-profile the candidate while looking for the
        # declared kernel in the CUDA trace. See `_collect` for why more than one is needed.
        if trace_attempts < 1:
            raise RunnerError("device proof needs at least one profiled attempt")
        self.trace_attempts = trace_attempts

    def ready(self) -> None:
        if self.device.type != "cuda" or self.device.index is None or not self.torch.cuda.is_initialized():
            raise RunnerError("fixture feedback requires the caller's already-initialized explicit CUDA device")

    def identity(self):
        self.ready()
        props = self.torch.cuda.get_device_properties(self.device)
        uuid = getattr(props, "uuid", None)
        return {"kind": "cuda", "device": str(self.device), "host": socket.gethostname(), "name": props.name,
                "uuid": str(uuid) if uuid is not None else None, "torch": self.torch.__version__,
                "triton": importlib.metadata.version("triton"), "capability": f"{props.major}.{props.minor}"}

    @contextmanager
    def execution(self):
        self.ready()
        with self.torch.cuda.device(self.device), self.torch.inference_mode():
            yield

    def time_us(self, call: Callable[[], Value]) -> float:
        self.ready()
        with self.torch.cuda.device(self.device):
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            self.torch.cuda.synchronize(self.device)
            start.record()
            call()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) * 1000

    def _collect(self, runner: ResidentRunner, observer: TritonObserver, invocation: LocalInvocation,
                 call, index: int):
        """Execute the candidate once under the profiler and report what the trace contained.

        Returns `(attempt, result, names, sample)` for the call the caller supplied, so the
        correctness and output-participation checks are always made against the value this attempt
        actually returned rather than a remembered one.

        Why this is a separate function that can be called more than once: the CUDA side of a
        `torch.profiler` trace has been observed to come back EMPTY while the CPU side is complete
        -- `cuLaunchKernelEx` present, `Activity Buffer Request` present, 11-12 CPU events -- so the
        kernel demonstrably launched and the profiler demonstrably ran, yet no device activity record
        arrived in time to be returned. Five separate probes ruled out the reconstructable causes (a
        profiling server, the candidate's bytes, the helper RPC path, a fresh profiler per session,
        and each layer of this context stack in turn), which leaves an asynchronous flush that
        sometimes misses its window. Under `require_device_proof` that empty tuple REJECTS a correct
        kernel, so a single read is not a sound basis for a verdict either way.

        Re-executing is legitimate here because this is a proof, not a measurement: latency comes
        from `local_eval`'s paired microbenchmark, never from these attempts, so running the
        candidate again cannot flatter or penalise a timing. It only changes how many chances the
        profiler gets to report activity that has already happened.
        """
        with self.profiler.profile(activities=[self.profiler.ProfilerActivity.CPU,
                                               self.profiler.ProfilerActivity.CUDA]) as trace:
            with runner.binding.diagnostic():
                result = invocation.candidate(call)
                self.torch.cuda.synchronize(self.device)
            counts = runner.binding.counts.get(invocation.fixture.original.module_path, {})
            if not counts.get("replacement_calls") or counts.get("old_calls"):
                raise RunnerError("selected replacement did not execute exclusively")
        # Read the events ONCE and OUTSIDE the profiler context: `trace.events()` returns an empty
        # list when called before the context exits its own step (measured: 0 events in 12/12
        # sessions inside, 7/7 with one CUDA record outside), so the read position is load-bearing.
        events = trace.events()
        cuda = tuple(e for e in events if str(e.device_type).endswith("CUDA"))
        names = tuple(e.name for e in cuda)
        every = tuple(e.name for e in events)
        sample = tuple(dict.fromkeys(every))[:8]
        kernel_name = str(getattr(observer.compiled, "name", ""))
        attempt = TraceAttempt(index=index, events=len(events), cuda_events=len(cuda),
            launch_calls=observer.counters["launch_calls"], kernel_seen=bool(kernel_name) and kernel_name in names,
            launch_api_seen=any(name in LAUNCH_API_EVENTS for name in every),
            names=sample, cuda_names=tuple(dict.fromkeys(names))[:8])
        return attempt, result, names, sample

    def prove(self, runner: ResidentRunner, declaration: DeviceKernelDeclaration,
              invocation: LocalInvocation) -> KernelEvidence:
        self.ready()
        if str(getattr(runner.backend, "device", "")) != str(self.device):
            raise RunnerError("fixture device must match the existing resident model device")
        if any(t.device != str(self.device) for t in invocation.fixture.description.tensors):
            raise RunnerError("mixed/CPU tensor fixtures are unsupported by the CUDA proof adapter")
        if declaration.backend != "triton":
            raise RunnerError("CUDA C++ device declarations are explicitly unsupported by this first adapter")
        observer = TritonObserver(runner, declaration, invocation)
        names: tuple[str, ...] = ()
        seen_events = seen_cuda = 0
        sample: tuple[str, ...] = ()
        attempts: list[TraceAttempt] = []
        output_used, suppressed = False, False
        normal_correct, quality_detail = True, None
        try:
            with self.torch.cuda.device(self.device), self.torch.inference_mode(), observer.observe():
                call = result = None
                for index in range(self.trace_attempts):
                    if call is not None:
                        del result, call
                    # Each attempt gets a fresh input clone AND a reset observer, so the tensors and
                    # provenance recorded below always belong to the attempt being judged. Without
                    # the reset, output pointers from a discarded attempt would still be in
                    # `output_tensors` and `output_matches` would report False for a kernel that did
                    # write its declared output.
                    call = self.tools.clone(invocation.fixture.original.call)
                    observer.begin_attempt()
                    observer.set_inputs(call)
                    attempt, result, names, sample = self._collect(runner, observer, invocation, call, index)
                    attempts.append(attempt)
                    seen_events, seen_cuda = attempt.events, attempt.cuda_events
                    # Stop as soon as the trace shows the declared kernel; a further attempt could
                    # only re-confirm it. Continuing after an empty read is what turns a missed
                    # flush into a retry instead of a rejection.
                    if attempt.kernel_seen:
                        break
                try:
                    self.tools.check(invocation.fixture.original.expected, result)
                    self.tools.check_state(invocation.fixture.original.touched_state, self.tools.state(call))
                except (AssertionError, RunnerError) as exc:
                    normal_correct, quality_detail = False, str(exc)
                output_used = observer.output_matches(result)
                del result, call
                if normal_correct:
                    observer.skip = True
                    control_call = self.tools.clone(invocation.fixture.original.call)
                    observer.set_inputs(control_call)
                    control = invocation.candidate(control_call)
                    self.torch.cuda.synchronize(self.device)
                    try:
                        self.tools.check(invocation.fixture.original.expected, control)
                    except (AssertionError, RunnerError):
                        suppressed = True
        finally:
            for key, value in observer.counters.items():
                self.counters[key] += value
        compiled = observer.compiled
        metadata = getattr(compiled, "metadata", None)
        return KernelEvidence(backend="triton", source_file=declaration.source_file, entry=declaration.entry,
            compiled=compiled is not None, kernel_name=str(getattr(compiled, "name", "")),
            launches=observer.counters["launch_calls"], cuda_names=names, output_used=output_used,
            stored_output_args=tuple(sorted(observer.stored_outputs)), loaded_input_args=tuple(sorted(observer.read_inputs)),
            skip_control_rejected=suppressed, compile_calls=observer.counters["compile_calls"],
            local_quality_passed=normal_correct, quality_detail=quality_detail,
            compile_wall_ms=observer.counters["compile_wall_us"] / 1000, jit_calls=observer.counters["jit_calls"],
            n_regs=getattr(compiled, "n_regs", None), shared_bytes=getattr(metadata, "shared", None),
            trace_events=seen_events, trace_cuda_events=seen_cuda, trace_sample=sample,
            trace_attempts=tuple(attempts))
