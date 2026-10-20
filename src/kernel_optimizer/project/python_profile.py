"""Isolated CPU call profiling around a warmed project; never a score sample."""
from pathlib import Path
import cProfile
import pstats
import time

from kernel_optimizer.runtime.execution import write_json


def capture(adapter, state, operation, synchronize, context):
    import torch
    adapter.reset(state)
    if callable(getattr(adapter, "before_run", None)):
        adapter.before_run(state)
    elif torch.distributed.is_initialized():
        torch.distributed.barrier()
    synchronize()
    profiler = cProfile.Profile()
    started = time.perf_counter_ns()
    with profiler:
        operation()
        synchronize()
    elapsed = (time.perf_counter_ns()-started)/1e6
    destination = Path(context["output"])/f"python-rank-{context['rank']}.pstats"
    profiler.dump_stats(str(destination))
    rows = []
    for (file, line, function), (primitive, calls, own, cumulative, callers) in pstats.Stats(profiler).stats.items():
        rows.append({"file": file, "line": line, "function": function, "calls": calls,
                     "primitive_calls": primitive, "self_ms": own*1000, "cumulative_ms": cumulative*1000})
    report = {"rank": context["rank"], "profiled_wall_ms": elapsed, "pstats": str(destination),
              "by_self_time": sorted(rows, key=lambda r: r["self_ms"], reverse=True)[:80],
              "by_cumulative_time": sorted(rows, key=lambda r: r["cumulative_ms"], reverse=True)[:80],
              "scope": "cProfile perturbs Python dispatch; self/cumulative times include host waits and cannot be added to GPU durations or used for acceptance"}
    path = destination.with_suffix(".json")
    write_json(path, report)
    return {"valid": True, "report": str(path), "collector": "python_runtime"}
