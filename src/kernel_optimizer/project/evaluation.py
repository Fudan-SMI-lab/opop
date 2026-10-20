"""Fixed evaluation mechanics around an agent-generated task adapter."""
from __future__ import annotations

import importlib.util
from contextlib import ExitStack
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("opop_generated_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("prepare", "reset", "run", "observe", "metrics", "models"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"adapter missing callable {name}")
    return module


def snapshot(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    raise TypeError(f"unsupported quality observation: {type(value).__name__}")


def compare(actual: Any, expected: Any, quality: dict, path: str = "output") -> dict:
    import torch
    maximum = 0.0
    count = 0
    current_path = path

    def visit(a, b, name):
        nonlocal maximum, count, current_path
        current_path = name
        if isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor):
                raise AssertionError(f"{name}: tensor structure changed")
            if b.numel() and a.shape == b.shape and (b.is_floating_point() or b.is_complex()):
                error = float((a.detach().cpu()-b).abs().max())
                maximum = max(maximum, error) if math.isfinite(error) and maximum is not None else None
            count += b.numel()
            if not b.is_floating_point() and not b.is_complex():
                if a.dtype != b.dtype or not torch.equal(a.detach().cpu(), b):
                    raise AssertionError(f"{name}: integer/bool tensors must match exactly")
            else:
                torch.testing.assert_close(a.detach().cpu(), b, atol=float(quality.get("atol", 0)),
                                           rtol=float(quality.get("rtol", 0)), equal_nan=False)
        elif isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                raise AssertionError(f"{name}: dictionary keys changed")
            for key in b:
                visit(a[key], b[key], f"{name}.{key}")
        elif isinstance(b, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                raise AssertionError(f"{name}: sequence structure changed")
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, f"{name}[{index}]")
        elif isinstance(b, float):
            if not isinstance(a, (int, float)) or not math.isfinite(a) or not math.isfinite(b) or not math.isclose(
                    a, b, abs_tol=quality.get("atol", 0), rel_tol=quality.get("rtol", 0)):
                raise AssertionError(f"{name}: scalar differs")
            count += 1
            if maximum is not None:
                maximum = max(maximum, abs(a-b))
        elif a != b or type(a) is not type(b):
            raise AssertionError(f"{name}: exact value differs")
        else:
            count += 1
    try:
        visit(actual, expected, path)
        return {"passed": True, "compared_elements": count, "max_abs_error": maximum}
    except AssertionError as exc:
        return {"passed": False, "compared_elements": count, "max_abs_error": maximum,
                "observation_path": current_path, "atol": quality.get("atol", 0), "rtol": quality.get("rtol", 0),
                "failure_scope": "first failing observation in comparison traversal, not earliest execution divergence",
                "error": str(exc)[:2000]}


def corrupt(value: Any) -> Any:
    import torch
    changed = False
    def visit(item):
        nonlocal changed
        if isinstance(item, torch.Tensor):
            copy = item.clone()
            if not changed and copy.numel():
                copy = copy.contiguous()
                if copy.dtype == torch.bool:
                    copy.view(-1)[0] = ~copy.view(-1)[0]
                elif copy.is_floating_point() or copy.is_complex():
                    copy.view(-1)[0] += copy.view(-1)[0].abs() * 10 + 1
                else:
                    copy.view(-1)[0] += 1
                changed = True
            return copy
        if isinstance(item, dict):
            return {key: visit(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(visit(v) for v in item)
        if not changed and isinstance(item, float):
            changed = True
            return item + abs(item)*10 + 1
        if not changed and isinstance(item, (int, bool)):
            changed = True
            return not item if isinstance(item, bool) else item+1
        return item
    result = visit(value)
    if not changed:
        raise ValueError("quality negative control needs a numeric observation")
    return result


def worker_entry(context: dict) -> dict:
    if context["parameters"].get("action") != "candidate_diagnostics":
        return _worker_entry(context)
    from kernel_optimizer.project.candidate_diagnostics import CandidateTrace
    from kernel_optimizer.runtime.execution import write_json
    trace = CandidateTrace(context["parameters"].get("bundles", []))
    try:
        with trace:
            result = _worker_entry(context, candidate_trace=trace)
        return {**result, "candidate_diagnostics": trace.report()}
    finally:
        write_json(Path(context["output"])/f"candidate-diagnostics-rank-{context['rank']}.json", trace.report())


def _worker_entry(context: dict, *, candidate_trace=None) -> dict:
    import torch
    parameters = context["parameters"]
    request = parameters["request"]
    action = parameters.get("action", "controls")
    ablation = parameters.get("collector_options", {}) if action == "binding_ablation" else {}
    passthrough = set(ablation.get("passthrough_bundles", []))
    if passthrough - {b["id"] for b in parameters.get("bundles", [])}:
        raise ValueError("diagnostic passthrough references an absent bundle")
    if ablation.get("direct") and parameters.get("bundles"):
        raise ValueError("direct adapter ablation requires an unmodified revision")
    adapter = load_adapter(Path(parameters["adapter"]))
    state = adapter.prepare(context, request)
    cuda = str(context["device"]).startswith("cuda")
    lifecycle_allocated_peak = torch.cuda.max_memory_allocated() if cuda else 0
    lifecycle_reserved_peak = torch.cuda.max_memory_reserved() if cuda else 0
    policies = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
    bindings = []
    setup = []
    for bundle in parameters.get("bundles", []):
        from kernel_optimizer.project.binding import BindingSession
        cuda = str(context["device"]).startswith("cuda")
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        binding = BindingSession(adapter, state, bundle["specs"], rank=context["rank"], world_size=context["world_size"], bundle_id=bundle["id"])
        binding.tracking = False
        installed = binding.install(Path(bundle["file"]), bundle.get("parameters", {}),
                                    diagnostic_passthrough=bundle["id"] in passthrough)
        expected = bundle.get("expected_calls_by_rank", {}).get(str(context["rank"]), bundle["expected_calls"])
        bundle = {**bundle, "expected_calls": expected}
        if set(installed) != set(expected):
            raise ValueError("installed candidate sites differ from the declared bundle")
        if cuda:
            torch.cuda.synchronize()
        setup.append({"bundle": bundle["id"], "elapsed_ms": (time.perf_counter_ns()-started)/1e6,
                      "allocated_bytes": torch.cuda.memory_allocated() if cuda else None,
                      "reserved_bytes": torch.cuda.memory_reserved() if cuda else None,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
                      "scope": "installation after model prepare; separate from the declared warm-model score"})
        if cuda:
            lifecycle_allocated_peak = max(lifecycle_allocated_peak, torch.cuda.max_memory_allocated())
            lifecycle_reserved_peak = max(lifecycle_reserved_peak, torch.cuda.max_memory_reserved())
        bindings.append((bundle, binding))
    if candidate_trace is not None:
        candidate_trace.phase = "run"
    quality = request["quality"]
    reference_path = Path(parameters.get("reference_dir", context["output"])) / f"reference-rank-{context['rank']}.pt"

    def mode(name):
        if callable(getattr(adapter, "set_mode", None)):
            adapter.set_mode(state, name)

    mode("quality")

    distributed = torch.distributed.is_initialized()

    def synchronize():
        if str(context["device"]).startswith("cuda"):
            devices = range(torch.cuda.device_count()) if context["world_size"] == 1 else [torch.cuda.current_device()]
            for device in devices:
                torch.cuda.synchronize(device)

    def consensus(local):
        if not distributed:
            return local
        flag = torch.tensor(int(local), device=context["device"], dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def operation(*, track=False):
        if ablation.get("direct"):
            return adapter.run(state)
        try:
            with ExitStack() as stack:
                for _, binding in bindings:
                    binding.tracking = track
                    if binding._mode is not None:
                        stack.enter_context(binding._mode)
                return adapter.run(state)
        finally:
            for _, binding in bindings:
                binding.tracking = False

    def invoke(*, track=False):
        nonlocal lifecycle_allocated_peak, lifecycle_reserved_peak
        if cuda:
            lifecycle_allocated_peak = max(lifecycle_allocated_peak, torch.cuda.max_memory_allocated())
            lifecycle_reserved_peak = max(lifecycle_reserved_peak, torch.cuda.max_memory_reserved())
        if callable(getattr(adapter, "before_run", None)):
            adapter.before_run(state)
        elif distributed:
            torch.distributed.barrier()
        synchronize()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        result = operation(track=track)
        synchronize()
        elapsed = (time.perf_counter_ns()-started)/1e6
        metrics = adapter.metrics(state, result, elapsed)
        if not isinstance(metrics, dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError("adapter metrics must be finite numeric values")
        metrics["framework.elapsed_ms"] = elapsed
        if cuda:
            lifecycle_allocated_peak = max(lifecycle_allocated_peak, torch.cuda.max_memory_allocated())
            lifecycle_reserved_peak = max(lifecycle_reserved_peak, torch.cuda.max_memory_reserved())
            metrics.update({"framework.peak_allocated_bytes": float(torch.cuda.max_memory_allocated()),
                            "framework.peak_reserved_bytes": float(torch.cuda.max_memory_reserved()),
                            "framework.allocated_bytes": float(torch.cuda.memory_allocated()),
                            "framework.reserved_bytes": float(torch.cuda.memory_reserved()),
                            "framework.lifecycle_peak_allocated_bytes": float(lifecycle_allocated_peak),
                            "framework.lifecycle_peak_reserved_bytes": float(lifecycle_reserved_peak)})
        return result, metrics

    def summarize(samples):
        metrics = {name: statistics.median(sample[name] for sample in samples) for name in samples[0]}
        metric_samples, diagnostics = {}, {}
        if request["objective"].get("source"):
            from kernel_optimizer.project.collectors import configured
            mode("profile")
            source = request["objective"]["source"]
            diagnostics = configured(request.get("profiling", {})).resolve(source["collector"]).metric(
                adapter, state, operation, synchronize, context, source, int(request.get("profiling", {}).get("metric_repeats", 3)))
            metric_samples[request["objective"]["name"]] = diagnostics["values"]
            metrics[request["objective"]["name"]] = statistics.median(diagnostics["values"])
        if request["objective"]["name"] not in metrics:
            raise ValueError(f"evaluator omitted objective {request['objective']['name']}")
        return {"metrics": metrics, "metric_samples": metric_samples, "diagnostics": diagnostics}

    if action == "candidate_diagnostics":
        adapter.reset(state)
        for _ in range(request["continuous_steps"]):
            invoke(track=True)
        return {"valid": True, "coverage": {
            bundle["id"]: {"counts": dict(binding.counts), "fallbacks": dict(binding.fallbacks),
                           **binding.path_telemetry()} for bundle, binding in bindings},
                "scope": "Diagnostic execution only; no quality or performance acceptance"}
    if action == "collective_calibration":
        from kernel_optimizer.project.calibration import capture as calibrate
        return calibrate(context, parameters["collector_options"])
    if action == "telemetry_profile":
        from kernel_optimizer.project.telemetry import capture as telemetry_capture
        mode("profile")
        for _ in range(request["warmup"]):
            adapter.reset(state); invoke()
        return telemetry_capture(adapter, state, operation, synchronize, context, request)
    if action == "nsys_profile":
        from kernel_optimizer.project.nsys import capture as nsys_capture
        mode("profile")
        for _, binding in bindings:
            binding.tracking = False
        for _ in range(request["warmup"]):
            adapter.reset(state); invoke()
        return nsys_capture(adapter, state, operation, synchronize)
    if action == "execution_timeline":
        from kernel_optimizer.project.timeline import capture
        mode("measure")
        for _ in range(request["warmup"]):
            adapter.reset(state); invoke()
        return capture(adapter, state, operation, synchronize, context)
    if action == "profile":
        mode("profile")
        from kernel_optimizer.project.discovery import capture
        from kernel_optimizer.runtime.execution import write_json
        for _, binding in bindings:
            binding.tracking = False
        for _ in range(request["warmup"]):
            adapter.reset(state); invoke()
        try:
            for _, binding in bindings:
                binding.diagnostic = True
            observed = capture(adapter, state, Path(context["output"]), context["rank"], operation, synchronize)
        finally:
            for _, binding in bindings:
                binding.diagnostic = False
        from kernel_optimizer.project.communication import probe, normalize, transports, available_fields
        import json
        policy = request.get("communication", "auto")
        manifest = probe(context, policy, request.get("tools"))
        observed["capabilities"] = manifest.model_dump(mode="json")
        applicable = policy != "off" and (context["world_size"] > 1 or torch.cuda.device_count() > 1)
        if applicable:
            observed["communication"] = normalize(json.loads(Path(observed["trace"]).read_text()), observed, context["rank"])
            observed["communication"]["transport"] = transports(Path(context["output"]).glob("nccl.*.log"))
            from kernel_optimizer.project.communication import bucket_inventory, attach_bucket_candidates
            attach_bucket_candidates(observed["communication"], bucket_inventory(adapter, state, observed))
        else:
            observed["communication"] = {"status": "not_applicable", "reason": "disabled" if policy == "off" else "single-device task"}
        if policy == "required":
            required = set(request.get("required_observations") or ["nccl_kernel_duration"])-{"cross_rank_collectives"}
            available = available_fields(observed["communication"]) | {field.name for field in manifest.fields if field.status == "observed"}
            if required-available:
                raise ValueError(f"required communication observations unavailable: {sorted(required-available)}")
        destination = Path(context["output"]) / f"discovery-rank-{context['rank']}.json"
        write_json(destination, observed)
        return {"valid": True, "discovery": str(destination), "coverage": observed["coverage"],
                "call_sites": len(observed["calls"]), "groups": len(observed["groups"])}
    if action in {"evaluate", "binding_ablation"}:
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        adapter.reset(state)
        checks = []
        for index in range(request["continuous_steps"]):
            result, _ = invoke(track=True)
            checks.append({"continuous_step": index, **compare(adapter.observe(state, result), reference[index], quality)})
        coverage = {bundle["id"]: {"counts": dict(binding.counts), "fallbacks": dict(binding.fallbacks),
                                  "required": bundle["expected_calls"], **binding.path_telemetry()} for bundle, binding in bindings}
        covered = all(set(bundle["expected_calls"]) <= set(binding.counts) for bundle, binding in bindings)
        unchanged = policies == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
        valid = consensus(all(check["passed"] for check in checks) and covered and unchanged)
        samples = []
        if valid:
            mode("measure")
            for _, binding in bindings:
                binding.tracking = False
            for _ in range(request["warmup"]):
                adapter.reset(state); invoke()
            for _ in range(request["repeats"]):
                adapter.reset(state); _, metrics = invoke(); samples.append(metrics)
        return {"valid": valid, "checks": checks, "coverage": coverage, "math_policy_unchanged": unchanged,
                "setup": setup,
                "samples": samples, **(summarize(samples) if samples else {"metrics": {}}),
                **({"diagnostic_ablation": {"options": ablation, "scope":
                    "Quality-checked diagnostic; no candidate acceptance. Passthrough omits candidate build and computation; measures site binding, not a deployable candidate."}}
                   if action == "binding_ablation" else {}),
                "bundle_hashes": context.get("revision", {}).get("bundle_hashes", {})}
    if action != "controls":
        raise ValueError(f"unsupported evaluator action: {action}")
    adapter.reset(state)
    reference = []
    for _ in range(request["continuous_steps"]):
        result, _ = invoke()
        reference.append(snapshot(adapter.observe(state, result)))
    torch.save(reference, reference_path)
    adapter.reset(state)
    reset_checks = []
    for index in range(request["continuous_steps"]):
        result, _ = invoke()
        reset_checks.append(compare(adapter.observe(state, result), reference[index], quality))
    if not consensus(all(check["passed"] for check in reset_checks)):
        return {"valid": False, "failure": "state_reset_or_reference_instability", "checks": reset_checks}
    original = adapter.run
    calls = 0
    def identity(state):
        nonlocal calls
        calls += 1
        return original(state)
    adapter.run = identity
    adapter.reset(state)
    result, _ = invoke()
    observed = snapshot(adapter.observe(state, result))
    identity_check = compare(observed, reference[0], quality)
    wrong_output = compare(corrupt(observed), reference[0], quality)
    identity_covered = calls > 0
    adapter.run = original
    calls = 0
    adapter.reset(state)
    result, _ = invoke()
    uncovered_quality = compare(adapter.observe(state, result), reference[0], quality)
    missing_coverage_rejected = calls == 0 and uncovered_quality["passed"]
    controls_valid = consensus(identity_check["passed"] and identity_covered and not wrong_output["passed"] and missing_coverage_rejected)
    if not controls_valid:
        return {"valid": False, "failure": "identity_or_negative_control_failed_on_a_rank",
                "identity": identity_check, "wrong_output": wrong_output, "coverage": missing_coverage_rejected}
    mode("measure")
    for _ in range(request["warmup"]):
        adapter.reset(state); invoke()
    samples = []
    for _ in range(request["repeats"]):
        adapter.reset(state)
        _, metrics = invoke()
        samples.append(metrics)
    metric_names = set(samples[0])
    if any(set(sample) != metric_names for sample in samples):
        raise ValueError("metric schema changes across repeated runs")
    return {"valid": controls_valid,
            "controls": {"reset": reset_checks, "identity": identity_check, "identity_covered": identity_covered,
                         "wrong_output_rejected": not wrong_output["passed"], "missing_coverage_rejected": missing_coverage_rejected},
            "samples": samples, **summarize(samples),
            "reference": str(reference_path), "models": list(adapter.models(state))}
