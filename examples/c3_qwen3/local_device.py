import importlib
import importlib.metadata
import socket
from collections.abc import Callable
from contextlib import contextmanager

from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from .device_records import KernelEvidence, LocalInvocation
from .model_runner import ResidentRunner
from .operator_types import Value
from .pure_tensors import TorchPureTools
from .runner_records import RunnerError
from .triton_observer import TritonObserver


class TorchLocalDevice:
    def __init__(self, device: str) -> None:
        self.torch = importlib.import_module("torch")
        self.profiler = importlib.import_module("torch.profiler")
        self.device = self.torch.device(device)
        self.tools = TorchPureTools()
        self.counters = {"jit_calls": 0, "compile_calls": 0, "launch_calls": 0, "compile_wall_us": 0}

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
        call = self.tools.clone(invocation.fixture.original.call)
        observer.set_inputs(call)
        names: tuple[str, ...] = ()
        output_used, suppressed = False, False
        normal_correct, quality_detail = True, None
        try:
            with self.torch.cuda.device(self.device), self.torch.inference_mode(), observer.observe():
                with self.profiler.profile(activities=[self.profiler.ProfilerActivity.CPU, self.profiler.ProfilerActivity.CUDA]) as trace:
                    with runner.binding.diagnostic():
                        result = invocation.candidate(call)
                        self.torch.cuda.synchronize(self.device)
                    counts = runner.binding.counts.get(invocation.fixture.original.module_path, {})
                    if not counts.get("replacement_calls") or counts.get("old_calls"):
                        raise RunnerError("selected replacement did not execute exclusively")
                names = tuple(e.name for e in trace.events() if str(e.device_type).endswith("CUDA"))
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
            n_regs=getattr(compiled, "n_regs", None), shared_bytes=getattr(metadata, "shared", None))
