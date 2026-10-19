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

    def visit(a, b, name):
        nonlocal maximum, count
        if isinstance(b, torch.Tensor):
            if not isinstance(a, torch.Tensor):
                raise AssertionError(f"{name}: tensor structure changed")
            if not b.is_floating_point() and not b.is_complex():
                if a.dtype != b.dtype or not torch.equal(a.detach().cpu(), b):
                    raise AssertionError(f"{name}: integer/bool tensors must match exactly")
            else:
                torch.testing.assert_close(a.detach().cpu(), b, atol=float(quality.get("atol", 0)),
                                           rtol=float(quality.get("rtol", 0)), equal_nan=False)
            if b.numel() and (b.is_floating_point() or b.is_complex()):
                maximum = max(maximum, float((a.detach().cpu()-b).abs().max()))
            count += b.numel()
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
            count += 1; maximum = max(maximum, abs(a-b))
        elif a != b or type(a) is not type(b):
            raise AssertionError(f"{name}: exact value differs")
        else:
            count += 1
    try:
        visit(actual, expected, path)
        return {"passed": True, "compared_elements": count, "max_abs_error": maximum}
    except AssertionError as exc:
        return {"passed": False, "compared_elements": count, "max_abs_error": maximum, "error": str(exc)[:2000]}


def corrupt(value: Any) -> Any:
    import torch
    changed = False
    def visit(item):
        nonlocal changed
        if isinstance(item, torch.Tensor):
            copy = item.clone()
            if not changed and copy.numel() and copy.is_floating_point():
                copy = copy.contiguous()
                copy.view(-1)[0] += copy.view(-1)[0].abs() * 10 + 1
                changed = True
            return copy
        if isinstance(item, dict):
            return {key: visit(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(visit(v) for v in item)
        if not changed and isinstance(item, float):
            changed = True
            return item + abs(item)*10 + 1
        return item
    result = visit(value)
    if not changed:
        raise ValueError("quality negative control needs a numeric observation")
    return result


def worker_entry(context: dict) -> dict:
    import torch
    parameters = context["parameters"]
    request = parameters["request"]
    adapter = load_adapter(Path(parameters["adapter"]))
    state = adapter.prepare(context, request)
    policies = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
    bindings = []
    for bundle in parameters.get("bundles", []):
        from kernel_optimizer.project.binding import BindingSession
        binding = BindingSession(adapter, state, bundle["specs"])
        installed = binding.install(Path(bundle["file"]), bundle.get("parameters", {}))
        if set(installed) != set(bundle["expected_calls"]):
            raise ValueError("installed candidate sites differ from the declared bundle")
        bindings.append((bundle, binding))
    quality = request["quality"]
    reference_path = Path(parameters.get("reference_dir", context["output"])) / f"reference-rank-{context['rank']}.pt"

    def operation():
        with ExitStack() as stack:
            for _, binding in bindings:
                if binding._mode is not None:
                    stack.enter_context(binding._mode)
            return adapter.run(state)

    def invoke():
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns()-started)/1e6
        metrics = adapter.metrics(state, result, elapsed)
        if not isinstance(metrics, dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError("adapter metrics must be finite numeric values")
        return result, metrics

    action = parameters.get("action", "controls")
    if action == "profile":
        from kernel_optimizer.project.discovery import capture
        from kernel_optimizer.runtime.execution import write_json
        for _, binding in bindings:
            binding.tracking = False
        for _ in range(request["warmup"]):
            adapter.reset(state); invoke()
        observed = capture(adapter, state, Path(context["output"]), context["rank"], operation)
        destination = Path(context["output"]) / f"discovery-rank-{context['rank']}.json"
        write_json(destination, observed)
        return {"valid": True, "discovery": str(destination), "coverage": observed["coverage"],
                "call_sites": len(observed["calls"]), "groups": len(observed["groups"])}
    if action == "evaluate":
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        adapter.reset(state)
        checks = []
        for index in range(request["continuous_steps"]):
            result, _ = invoke()
            checks.append(compare(adapter.observe(state, result), reference[index], quality))
        coverage = {bundle["id"]: {"counts": dict(binding.counts), "fallbacks": dict(binding.fallbacks),
                                  "required": bundle["expected_calls"]} for bundle, binding in bindings}
        covered = all(set(bundle["expected_calls"]) <= set(binding.counts) for bundle, binding in bindings)
        unchanged = policies == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
        valid = all(check["passed"] for check in checks) and covered and unchanged
        samples = []
        if valid:
            for _, binding in bindings:
                binding.tracking = False
            for _ in range(request["warmup"]):
                adapter.reset(state); invoke()
            for _ in range(request["repeats"]):
                adapter.reset(state); _, metrics = invoke(); samples.append(metrics)
        return {"valid": valid, "checks": checks, "coverage": coverage, "math_policy_unchanged": unchanged,
                "samples": samples, "metrics": {name: statistics.median(s[name] for s in samples)
                                                 for name in samples[0]} if samples else {},
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
    if not all(check["passed"] for check in reset_checks):
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
    for _ in range(request["warmup"]):
        adapter.reset(state); invoke()
    samples = []
    for _ in range(request["repeats"]):
        adapter.reset(state)
        _, metrics = invoke()
        samples.append(metrics)
    primary = request["objective"]["name"]
    if any(primary not in sample for sample in samples):
        raise ValueError(f"adapter omitted requested objective {primary}")
    metric_names = set(samples[0])
    if any(set(sample) != metric_names for sample in samples):
        raise ValueError("metric schema changes across repeated runs")
    return {"valid": identity_check["passed"] and identity_covered and not wrong_output["passed"] and missing_coverage_rejected,
            "controls": {"reset": reset_checks, "identity": identity_check, "identity_covered": identity_covered,
                         "wrong_output_rejected": not wrong_output["passed"], "missing_coverage_rejected": missing_coverage_rejected},
            "samples": samples, "metrics": {name: statistics.median(s[name] for s in samples) for name in metric_names},
            "reference": str(reference_path), "models": list(adapter.models(state))}
