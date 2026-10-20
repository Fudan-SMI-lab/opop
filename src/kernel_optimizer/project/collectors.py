"""Replaceable diagnostic and metric collectors, selected by request configuration."""
import importlib
import shutil
import subprocess
from uuid import uuid4

from kernel_optimizer.project.ports import Registry


class TorchCommunication:
    metric_fields = {
        "communication_kernel_sum_ms": {
            "unit": "ms", "meaning": "Per-rank sum of device communication-kernel durations in a separately profiled invocation; includes waiting, not wire time or a critical-path lower bound."},
        "communication_union_ms": {
            "unit": "ms", "meaning": "Per-rank union of communication-kernel intervals."},
        "compute_overlap_ms": {
            "unit": "ms", "meaning": "Local trace-clock overlap of communication and compute intervals."},
        "communication_without_compute_ms": {
            "unit": "ms", "meaning": "Communication intervals without observed compute overlap; not a measured critical path."},
    }

    def metric(self, adapter, state, operation, synchronize, context, source, repeats):
        from kernel_optimizer.project.communication import sample_metric
        return sample_metric(adapter, state, operation, synchronize, context, source, repeats)


class NcclTuning:
    def collect(self, search, revision):
        from kernel_optimizer.project.nccl_tuning import read_logs
        if search.request.communication == "off" or len(search.request.devices) < 2:
            return {"status": "not_applicable", "reason": "communication disabled or single-device task"}
        name = "nccl-tuning-" + uuid4().hex[:10]
        run = search.execute(revision, name, "nccl_tuning")
        result = read_logs((search.output/"jobs"/name/"run").glob("nccl.*.log"))
        return {**result, "run": run, "status": result["status"] if run["status"] == "passed" else "unavailable"}


class Nsys:
    def collect(self, search, revision):
        from kernel_optimizer.project.nsys import export
        name = "nsys-"+uuid4().hex[:10]
        run = search.execute(revision, name, "nsys_profile")
        counter_probe = {"status": "not_requested"}
        if search.request.profiling.get("gpu_counters", False):
            logs = list((search.output/"jobs"/name/"run").glob("rank-*.log"))
            denied = any("ERR_NVGPUCTRPERM" in path.read_text(errors="replace") for path in logs)
            counter_probe = {"status": "permission_denied" if denied else "attempted",
                             "evidence": [str(path) for path in logs], "run": run}
            if run["status"] != "passed":
                # Optional counters must not suppress the independently available trace.
                name = "nsys-"+uuid4().hex[:10]
                run = search.execute(revision, name, "nsys_profile", collector_options={"gpu_counters": False})
        if run["status"] != "passed":
            return {"status": "unavailable", "run": run, "counter_probe": counter_probe}
        tool = search.request.tools.get("nsys") or shutil.which("nsys")
        rows = export(tool, search.output/"jobs"/name/"run", [rank["rank"] for rank in run["ranks"]])
        return {"status": "observed" if all(row["status"] == "observed" for row in rows) else "unavailable",
                "ranks": rows, "run": run, "counter_probe": counter_probe}


class DeviceTelemetry:
    def collect(self, search, revision):
        if search.request.communication == "off":
            return {"status": "not_applicable", "reason": "communication diagnostics disabled"}
        run = search.execute(revision, "telemetry-"+uuid4().hex[:10], "telemetry_profile")
        rows = [r.get("result", {}).get("telemetry", {"status": "unavailable"}) for r in run.get("ranks", [])]
        return {"status": "observed" if run["status"] == "passed" and rows and all(r["status"] == "observed" for r in rows) else "unavailable",
                "ranks": rows, "run": run, "attribution": "device diagnostic window; no per-group traffic attribution"}


class CollectiveCalibration:
    def collect(self, search, revision):
        import json
        from kernel_optimizer.project.calibration import layouts
        if search.request.communication == "off":
            return {"status": "not_applicable", "reason": "communication diagnostics disabled"}
        discovery = json.loads((search.output/"discovery.json").read_text())
        options = layouts(discovery, len(search.request.devices))
        if not options["layouts"]:
            return {"status": "unavailable", **options}
        options.update(warmup=search.request.profiling.get("calibration_warmup", 2),
                       repeats=search.request.profiling.get("calibration_repeats", 5))
        run = search.execute(revision, "calibration-"+uuid4().hex[:10], "collective_calibration", collector_options=options)
        rows = [r.get("result", {}).get("calibration", {"status": "unavailable"}) for r in run.get("ranks", [])]
        return {"status": "observed" if run["status"] == "passed" and rows and all(r["status"] == "observed" for r in rows) else "unavailable",
                "ranks": rows, "run": run, "protocol": options}


class CandidateDiagnostics:
    def collect(self, search, revision):
        import json
        if not revision.bundles:
            return {"status": "not_applicable", "reason": "revision has no candidate installers"}
        name = "candidate-diagnostics-"+uuid4().hex[:10]
        run = search.execute(revision, name, "candidate_diagnostics")
        files = sorted((search.output/"jobs"/name/"run").glob("candidate-diagnostics-rank-*.json"))
        return {"status": "observed" if run["status"] == "passed" else "partial" if files else "unavailable",
                "run": run, "evidence": [str(p) for p in files],
                "ranks": [{"file": str(p), **json.loads(p.read_text())} for p in files]}


class ExecutionTimeline:
    def collect(self, search, revision):
        import json
        from pathlib import Path
        from kernel_optimizer.project.timeline import cross_rank
        options = search.request.profiling.get("execution_timeline", {})
        run = search.execute(revision, "timeline-"+uuid4().hex[:10], "execution_timeline",
                             collector_options=options)
        ranks = []
        for rank in run.get("ranks", []):
            ranks.extend(json.loads(Path(p).read_text()) for p in rank.get("result", {}).get("reports", []))
        return {"status": "observed" if run["status"] == "passed" and ranks else "unavailable",
                "ranks": ranks, "cross_rank": cross_rank(ranks), "run": run}


class PythonRuntime:
    def collect(self, search, revision):
        import json
        from pathlib import Path
        run = search.execute(revision, "python-runtime-"+uuid4().hex[:10], "python_runtime_profile")
        ranks = [json.loads(Path(r["result"]["report"]).read_text()) for r in run.get("ranks", [])
                 if r.get("result", {}).get("report")]
        return {"status": "observed" if run["status"] == "passed" and ranks else "unavailable",
                "ranks": ranks, "run": run}


def configured(profiling):
    registry = Registry()
    registry.register("torch_communication", TorchCommunication())
    registry.register("nccl_tuning", NcclTuning())
    registry.register("nsys", Nsys())
    registry.register("device_telemetry", DeviceTelemetry())
    registry.register("collective_calibration", CollectiveCalibration())
    registry.register("candidate_diagnostics", CandidateDiagnostics())
    registry.register("execution_timeline", ExecutionTimeline())
    registry.register("python_runtime", PythonRuntime())
    for name, factory in profiling.get("collector_plugins", {}).items():
        module, attribute = factory.split(":", 1)
        registry.register(name, getattr(importlib.import_module(module), attribute)())
    return registry


def diagnostic_names(profiling):
    return ["torch", "nsys", "nccl_tuning", "device_telemetry", "collective_calibration", "candidate_diagnostics", "execution_timeline", "python_runtime",
            *profiling.get("collector_plugins", {})]


def metric_catalog(profiling):
    """Describe installed metric interfaces, without claiming hardware availability."""
    return {name: {"fields": getattr(provider, "metric_fields", None),
                   "availability": "must be verified by the actual workload worker"}
            for name, provider in configured(profiling).items() if callable(getattr(provider, "metric", None))}


def collect(search, revision, names):
    import hashlib
    from kernel_optimizer.runtime.provenance import framework_source
    registry, results = configured(search.request.profiling), {}
    for name in names:
        if name == "torch":
            continue  # The coordinator's execution discovery already owns this run.
        try:
            results[name] = registry.resolve(name).collect(search, revision)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            results[name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        results[name]["provenance"] = {
            "revision": revision.id, "parent_revision": revision.parent_id,
            "bundles": list(revision.bundles), "plan": revision.plan_ref, "registry": revision.registry_ref,
            "request_sha256": hashlib.sha256((search.output/"request.json").read_bytes()).hexdigest(),
            "objective": search.request.objective, "workload": search.request.workload,
            "devices": search.request.devices, "framework_source": framework_source(),
            "scope": "Separate diagnostic on this exact version; availability and perturbation are collector-specific"}
    return results
