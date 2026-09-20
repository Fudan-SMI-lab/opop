import importlib
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
from .device_ir import stored_dependencies
from .device_records import LocalInvocation
from .model_runner import ResidentRunner
from .pure_tensors import tensor_leaves
from .runner_records import RunnerError


class TritonObserver:
    def __init__(self, runner: ResidentRunner, declaration: DeviceKernelDeclaration, invocation: LocalInvocation) -> None:
        imported = runner.binding.imported
        if imported is None or declaration.source_file not in invocation.bundle.sources:
            raise RunnerError("declared device source is not in the loaded cumulative bundle")
        module_name = ".".join((imported.namespace, *Path(declaration.source_file).with_suffix("").parts))
        module = importlib.import_module(module_name)
        kernel = module
        for part in declaration.entry.split("."):
            kernel = getattr(kernel, part, None)
            if kernel is None:
                raise RunnerError(f"declared kernel entry absent: {declaration.entry}")
        jit_type = importlib.import_module("triton.runtime.jit").JITFunction
        if not isinstance(kernel, jit_type):
            raise RunnerError("unsupported declaration: entry must resolve to an actual Triton JITFunction")
        source = inspect.getsourcefile(kernel.fn)
        if source is None or Path(source).resolve() != (imported.root / declaration.source_file).resolve():
            raise RunnerError("declared kernel function belongs to a different source file")
        if not declaration.output_arg_indices:
            raise RunnerError("output_arg_indices required for direct-output device participation proof")
        self.kernel, self.declaration, self.invocation = kernel, declaration, invocation
        self.compiled_type = importlib.import_module("triton.compiler").CompiledKernel
        self.torch = importlib.import_module("torch")
        self.skip = False
        self.compiled = None
        self.output_tensors = []
        self.stored_outputs: set[int] = set()
        self.read_inputs: set[int] = set()
        self.counters = {"jit_calls": 0, "compile_calls": 0, "launch_calls": 0, "compile_wall_us": 0}
        module = runner.binding.modules[invocation.fixture.original.module_path]
        self.weight_ptrs = {p.data_ptr() for _, p in module.named_parameters()}
        self.input_ptrs: set[int] = set()

    @contextmanager
    def observe(self) -> Iterator[None]:
        kernel = self.kernel
        old_run, old_compile = kernel.run, kernel._do_compile
        owned_run, owned_compile = "run" in vars(kernel), "_do_compile" in vars(kernel)
        def compile_kernel(*args, **kwargs):
            start = perf_counter()
            self.counters["compile_calls"] += 1
            try:
                return old_compile(*args, **kwargs)
            finally:
                self.counters["compile_wall_us"] += int((perf_counter() - start) * 1e6)
        def run(*args, **kwargs):
            self.counters["jit_calls"] += 1
            parameters = {k: v for k, v in kwargs.items() if k in kernel.signature.parameters}
            bound = kernel.signature.bind_partial(*args, **parameters)
            bound.apply_defaults()
            arguments = [bound.arguments[p.name] for p in kernel.params]
            outputs = []
            for index in self.declaration.output_arg_indices:
                if index >= len(arguments) or not self.torch.is_tensor(arguments[index]):
                    raise RunnerError("declared output operand is not a tensor")
                output = arguments[index]
                if output.data_ptr() in self.weight_ptrs | self.input_ptrs:
                    raise RunnerError("unsupported in-place output aliases immutable fixture inputs/weights")
                outputs.append(output)
            if self.skip:
                for output in outputs:
                    if not self.torch.is_floating_point(output):
                        raise RunnerError("skip control supports floating output operands only")
                    output.fill_(float("nan"))
                return old_run(*args, **{**kwargs, "warmup": True})
            compiled = old_run(*args, **kwargs)
            if not isinstance(compiled, self.compiled_type):
                raise RunnerError("Triton run did not return an actual CompiledKernel")
            self.compiled = compiled
            if not kwargs.get("warmup", False):
                self.counters["launch_calls"] += 1
                self.output_tensors.extend(outputs)
                argument_indices = {p.name: i for i, p in enumerate(kernel.params) if not p.is_constexpr}
                provenance = stored_dependencies(compiled.asm.get("ttir", ""), argument_indices)
                inputs = {i for i, a in enumerate(arguments) if self.torch.is_tensor(a)
                          and a.data_ptr() in self.input_ptrs | self.weight_ptrs}
                for index in self.declaration.output_arg_indices:
                    reads = provenance.get(index, frozenset()) & inputs
                    if reads:
                        self.stored_outputs.add(index)
                        self.read_inputs.update(reads)
            return compiled
        kernel.run, kernel._do_compile = run, compile_kernel
        try:
            yield
        finally:
            if owned_run:
                kernel.run = old_run
            else:
                delattr(kernel, "run")
            if owned_compile:
                kernel._do_compile = old_compile
            else:
                delattr(kernel, "_do_compile")

    def set_inputs(self, call) -> None:
        self.input_ptrs = {getattr(t, "data_ptr")() for _, t in tensor_leaves((call.args, dict(call.kwargs)))}

    def output_matches(self, output) -> bool:
        returned = {getattr(t, "data_ptr")() for _, t in tensor_leaves(output)}
        return bool(self.output_tensors) and all(t.data_ptr() in returned for t in self.output_tensors)
