"""Replaceable diagnostic and metric collectors, selected by request configuration."""
import importlib
import shutil
import subprocess
from uuid import uuid4

from kernel_optimizer.project.ports import Registry


class TorchCommunication:
    def metric(self, adapter, state, operation, synchronize, context, source, repeats):
        from kernel_optimizer.project.communication import sample_metric
        return sample_metric(adapter, state, operation, synchronize, context, source, repeats)


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


def configured(profiling):
    registry = Registry()
    registry.register("torch_communication", TorchCommunication())
    registry.register("nsys", Nsys())
    for name, factory in profiling.get("collector_plugins", {}).items():
        module, attribute = factory.split(":", 1)
        registry.register(name, getattr(importlib.import_module(module), attribute)())
    return registry


def diagnostic_names(profiling):
    return ["torch", "nsys", *profiling.get("collector_plugins", {})]


def collect(search, revision, names):
    registry, results = configured(search.request.profiling), {}
    for name in names:
        if name == "torch":
            continue  # The coordinator's execution discovery already owns this run.
        try:
            results[name] = registry.resolve(name).collect(search, revision)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            results[name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return results
