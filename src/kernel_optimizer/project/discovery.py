"""Observed eager modules and functional dispatch, with explicit partial coverage.

Profiling is a separate invocation, never a substitute for clean project timing.
See PyTorch profiler and extending/Modes documentation; installed behavior is
verified by the captured raw trace and correlation coverage.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
import weakref

from kernel_optimizer.project.transactions import identity

_phase = ContextVar("project_phase", default="run")


@contextmanager
def phase(name):
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


def tensor_description(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
                "device": value.device.type, "requires_grad": value.requires_grad}
    if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
        return {"type": type(value).__qualname__, "value": str(value)}
    if isinstance(value, (list, tuple)):
        return [tensor_description(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_description(v) for k, v in value.items()}
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    return {"type": type(value).__qualname__}


def tensor_leaves(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)


from kernel_optimizer.project.trace import associate as _associate


def capture(adapter, state, output: Path, rank: int, operation=None, synchronize=None) -> dict:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    synchronize = synchronize or (torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)
    calls = {}
    events = []
    edges = set()
    handles = []
    producer = {}
    module_stack = []
    open_modules = defaultdict(list)

    def enter(target, kind, semantics, args, kwargs, attributes=None):
        signature = {"args": tensor_description(args), "kwargs": tensor_description(kwargs or {})}
        key = target + "|" + _phase.get()
        site_parent = module_stack[-1] if kind == "functional" and module_stack else None
        if kind == "functional":
            key += "|" + str(site_parent) + "|" + identity(signature)[:12]
        call_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        if call_id not in calls:
            calls[call_id] = {"id": call_id, "target": target, "kind": kind, "phase": _phase.get(),
                              "semantics": semantics, "attributes": attributes or {}, "signatures": [],
                              "count": 0, "parents": [], "site_parent": site_parent}
        call = calls[call_id]
        if signature not in call["signatures"]:
            call["signatures"].append(signature)
        call["count"] += 1
        if module_stack and module_stack[-1] not in call["parents"]:
            call["parents"].append(module_stack[-1])
        sequence = len(events)
        events.append({"sequence": sequence, "call_id": call_id})
        marker = torch.profiler.record_function("opop::"+str(sequence)); marker.__enter__()
        return call_id, marker

    def dependency_inputs(args, call_id):
        for tensor in tensor_leaves(args):
            previous = producer.get(id(tensor))
            if previous and previous[0]() is tensor and previous[1] != call_id:
                edges.add((previous[1], call_id, "data"))

    def dependency_outputs(outputs, call_id):
        for tensor in tensor_leaves(outputs):
            producer[id(tensor)] = (weakref.ref(tensor), call_id)

    class CaptureMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            call_id, marker = enter(str(func), "functional", str(getattr(func, "_schema", func)), args, kwargs)
            dependency_inputs((args, kwargs), call_id)
            try:
                result = func(*args, **(kwargs or {}))
                dependency_outputs(result, call_id)
                return result
            finally:
                marker.__exit__(None, None, None)

    seen = set()
    for root_name, model in adapter.models(state).items():
        for name, module in model.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            target = root_name + ("." + name if name else "")
            semantics = type(module).__module__ + "." + type(module).__qualname__
            attributes = {key: tensor_description(value) for key, value in vars(module).items()
                          if not key.startswith("_") and isinstance(value, (str, int, float, bool, tuple))}
            # Parameter values are per-instance data; shared implementation groups retain each instance.
            attributes["parameter_shapes"] = {key: tensor_description(value) for key, value in module.named_parameters(recurse=False)}
            def before(mod, args, kwargs, target=target, semantics=semantics, attributes=attributes):
                call_id, marker = enter(target, "module", semantics, args, kwargs, attributes)
                open_modules[id(mod)].append((call_id, marker)); module_stack.append(call_id)
                dependency_inputs((args, kwargs), call_id)
            def after(mod, args, kwargs, result):
                if open_modules[id(mod)]:
                    call_id, marker = open_modules[id(mod)].pop()
                    dependency_outputs(result, call_id)
                    module_stack.pop(); marker.__exit__(None, None, None)
            handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
    adapter.reset(state)
    if callable(getattr(adapter, "before_run", None)):
        adapter.before_run(state)
    elif torch.distributed.is_initialized():
        torch.distributed.barrier()
    synchronize()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,
                                    record_shapes=True, profile_memory=False) as prof:
            with CaptureMode():
                call_id, marker = enter("project.invoke", "project", "complete workload", (), {})
                module_stack.append(call_id)
                try:
                    (operation or (lambda: adapter.run(state)))()
                    synchronize()
                finally:
                    module_stack.pop(); marker.__exit__(None, None, None)
        trace_path = output / f"trace-rank-{rank}.json"
        prof.export_chrome_trace(str(trace_path))
    finally:
        for handle in handles:
            handle.remove()
    trace = json.loads(trace_path.read_text())
    association = _associate(trace, events, calls)
    groups = defaultdict(list)
    for call in calls.values():
        signature = {"kind": call["kind"], "semantics": call["semantics"],
                     "attributes": call["attributes"], "signatures": call["signatures"]}
        groups[identity(signature)[:20]].append(call["id"])
    return {"schema_version": 2, "site_identity": "functional context + phase + signature; module path + phase",
            "rank": rank, "calls": list(calls.values()), "events": events,
            "dependencies": [{"producer": a, "consumer": b, "kind": kind} for a, b, kind in sorted(edges)],
            "groups": [{"id": ident, "calls": values} for ident, values in groups.items()],
            "trace": str(trace_path), "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "unknown_dependencies": ["Alias, persistent state and cross-stream/collective dependencies need additional evidence; data edges alone do not authorize reordering"],
            **association}
