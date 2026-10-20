"""Passive, separately measured execution evidence; no search-policy routing."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

from kernel_optimizer.project.communication import intervals, overlap, normalize, trace_window
from kernel_optimizer.runtime.execution import write_json


def _event(row):
    return {"name": row.get("name"), "start_us": row["ts"],
            "duration_us": row.get("dur", 0), "pid": row.get("pid"),
            "tid": row.get("tid"), "args": row.get("args", {})}


def clock_mapping(trace, anchors, domain):
    """Bracket trace CPU markers with the host monotonic clock, never wall time."""
    bounds = []
    for anchor in anchors:
        rows = [r for r in trace.get("traceEvents", [])
                if r.get("ph") == "X" and r.get("name") == anchor["name"]]
        if len(rows) != 1:
            continue
        row = rows[0]
        bounds.append([anchor["before_ns"]/1000-row["ts"],
                       anchor["after_ns"]/1000-row["ts"]])
    if len(bounds) < 2 or max(b[0] for b in bounds) > min(b[1] for b in bounds):
        return {"status": "unknown", "reason": "missing or inconsistent bracketed CPU clock anchors",
                "anchor_bounds_us": bounds, "domain": domain}
    # Use the hull, not an unjustifiably precise average, retaining anchor jitter.
    return {"status": "observed", "offset_bounds_us": [min(b[0] for b in bounds), max(b[1] for b in bounds)],
            "anchor_bounds_us": bounds, "domain": domain,
            "scope": "CPU trace to host monotonic clock within this capture; excludes unknown GPU clock calibration error"}


def analyze(trace, *, rank, anchors=(), domain=None, gradients=None, inventory=None):
    raw_trace = trace
    trace, window = trace_window(trace, "opop.timeline_workload")
    rows = [r for r in trace.get("traceEvents", []) if r.get("ph") == "X" and "ts" in r]
    launches = defaultdict(list)
    external = defaultdict(list)
    cpu, copies, sync, devices = [], [], [], set()
    kernels = [r for r in rows if r.get("cat") == "kernel"]
    for row in rows:
        args = row.get("args", {})
        if row.get("cat") in {"cuda_runtime", "cuda_driver"}:
            if "correlation" in args:
                launches[args["correlation"]].append(row)
            if any(s in row.get("name", "").lower() for s in ("synchronize", "waitevent")):
                sync.append(_event(row))
        if row.get("cat") == "cpu_op":
            cpu.append(row)
            if "External id" in args:
                external[args["External id"]].append(row)
        if row.get("cat") in {"gpu_memcpy", "gpu_memset"}:
            copies.append(_event(row))
    communication = normalize(trace, {"kernels": []}, rank)
    for collective in communication["collectives"]:
        submit, api, links = {}, {}, []
        for kernel in collective["kernels"]:
            linked = launches.get(kernel["correlation"], [])
            if len(linked) != 1:
                links.append({"correlation": kernel["correlation"], "status": "unknown",
                              "reason": "CUDA launch correlation absent or ambiguous"})
                continue
            launch = linked[0]
            api[id(launch)] = _event(launch)
            owners = [r for r in external.get(launch.get("args", {}).get("External id"), [])
                      if "c10d" in r["name"].lower()]
            if not owners:
                # Same-thread lexical containment is weaker than an explicit ID.
                owners = [r for r in cpu if "c10d" in r["name"].lower()
                          and r.get("pid") == launch.get("pid") and r.get("tid") == launch.get("tid")
                          and r["ts"] <= launch["ts"] < r["ts"]+r.get("dur", 0)]
                method = "same-thread enclosing c10d range"
            else:
                method = "native External id"
            # Keep all ranges, including wait ranges; do not guess a unique op.
            for owner in owners:
                submit[id(owner)] = _event(owner)
            links.append({"correlation": kernel["correlation"], "status": "observed",
                          "cpu_association": method if owners else "unknown",
                          "launch_to_kernel_start_us": kernel["start_us"]-launch["ts"],
                          "interpretation": "enqueue-to-start interval includes scheduling/dependencies; not pure transport or GPU input readiness"})
        collective["cpu_ranges"] = sorted(submit.values(), key=lambda r: r["start_us"])
        collective["cuda_launches"] = sorted(api.values(), key=lambda r: r["start_us"])
        scopes = [r for r in collective["cpu_ranges"] if "wait" not in r["name"].lower()]
        # Native dispatch can produce nested ranges with the same c10d name.
        # Select a uniquely innermost enclosing submission, retaining all raw
        # wrappers. Crossing ranges or several launches with distinct submits
        # remain ambiguous.
        inner = [r for r in scopes if not any(
            q is not r and q["pid"] == r["pid"] and q["tid"] == r["tid"]
            and r["start_us"] <= q["start_us"]
            and q["start_us"]+q["duration_us"] <= r["start_us"]+r["duration_us"]
            and (q["start_us"], q["duration_us"]) != (r["start_us"], r["duration_us"])
            for q in scopes)]
        collective["cpu_submission"] = (
            {"status": "observed", **inner[0], "method": "unique innermost c10d range enclosing correlated CUDA launch"}
            if len(inner) == 1 else {"status": "unknown", "reason": "no unique enclosing CPU submission"})
        collective["launch_links"] = links
        collective["gpu_input_ready"] = {"status": "unknown", "reason": "CPU submission does not establish GPU producer completion"}
        collective["consumer_completion"] = {"status": "unknown", "reason": "no unique data-consumer/event dependency proven"}
    per_device = {}
    for row in kernels:
        devices.add(row.get("args", {}).get("device"))
    for device in devices:
        owned = [r for r in kernels if r.get("args", {}).get("device") == device]
        comm = [(r["ts"], r["ts"]+r.get("dur", 0)) for r in owned
                if "Collective name" in r.get("args", {}) or "nccl" in r.get("name", "").lower()]
        comp = [(r["ts"], r["ts"]+r.get("dur", 0)) for r in owned
                if not ("Collective name" in r.get("args", {}) or "nccl" in r.get("name", "").lower())]
        c, k = intervals(comm), intervals(comp)
        all_active = intervals(comm+comp)
        end_compute = max((v[1] for v in k), default=None)
        tail = None if end_compute is None else sum(max(0, end-max(start, end_compute)) for start, end in c)
        span = all_active[-1][1]-all_active[0][0] if all_active else 0
        per_device[str(device)] = {
            "kernel_count": len(owned), "kernel_span_us": span,
            "kernel_active_union_us": sum(b-a for a, b in all_active),
            "gaps_between_kernels_us": span-sum(b-a for a, b in all_active),
            "communication_intervals_us": c, "compute_intervals_us": k,
            "compute_overlap_us": overlap(c, k),
            "communication_after_last_compute_us": tail,
            "interpretation": "Observed kernel intervals only; copies and CPU work are separate. Gaps/tail are not recoverable time or a causal critical path."}
    readiness = []
    for name, metadata in (gradients or {}).items():
        for row in rows:
            if row.get("name") == name:
                readiness.append({**_event(row), "parameters": metadata})
    buckets = []
    for bucket in (inventory or {}).get("buckets", []):
        names = set(bucket["parameters"])
        seen = {name for event in readiness for name in event["parameters"] if name in names}
        occurrences = {name: [r["start_us"] for r in readiness if name in r["parameters"]] for name in sorted(names)}
        buckets.append({**bucket, "observed_parameter_count": len(seen),
                        "cpu_post_accumulate_times_us": occurrences,
                        "last_cpu_post_accumulate_us": max((t for ts in occurrences.values() for t in ts), default=None)
                            if seen == names and names else None,
                        "timed_collective_identity": "unknown",
                        "interpretation": "Passive reducer membership and CPU callbacks; repeated microsteps are retained. Not GPU bucket-ready timestamps."})
    return {"schema_version": 1, "rank": rank, "communication": communication,
            "devices": per_device, "cuda_synchronization": sync, "copies_and_memsets": copies,
            "gradient_readiness": {"status": "observed" if readiness else "not_observed",
                                   "events": readiness, "buckets": buckets,
                                   "scope": "CPU post-accumulate callbacks; no GPU synchronization or communication-hook replacement"},
            "clock": clock_mapping(raw_trace, anchors, domain), "workload_window": window,
            "scope": "Separate diagnostic capture, never the clean project objective"}


def cross_rank(reports):
    """Compare matched CPU submit ranges only when a shared clock is evidenced."""
    groups = defaultdict(list)
    for report in reports:
        for item in report["communication"]["collectives"]:
            groups[(item["process_group"], item["sequence"], item["operation"], item["participants"])].append((report, item))
    result = []
    for key, members in groups.items():
        row = {"process_group": key[0], "sequence": key[1], "operation": key[2],
               "status": "unknown", "reason": "participants, CPU submission or shared clock evidence missing"}
        result.append(row)
        try:
            participants = set(json.loads(key[3]))
        except (TypeError, ValueError):
            continue
        ranks = [r["rank"] for r, _ in members]
        if not participants or set(ranks) != participants or len(ranks) != len(participants):
            continue
        clocks = [r["clock"] for r, _ in members]
        domains = [c.get("domain") for c in clocks]
        if (any(c.get("status") != "observed" for c in clocks) or not domains[0]
                or not domains[0].get("boot_id") or any(d != domains[0] for d in domains)):
            continue
        bounds = {}
        for report, item in members:
            submission = item.get("cpu_submission", {})
            if submission.get("status") != "observed":
                break
            ts = submission["start_us"]
            bounds[str(report["rank"])] = [ts+offset for offset in report["clock"]["offset_bounds_us"]]
        if len(bounds) != len(participants):
            continue
        lows = [v[0] for v in bounds.values()]
        highs = [v[1] for v in bounds.values()]
        row.update(status="observed", reason=None, cpu_submit_monotonic_bounds_us=bounds,
                   cpu_submit_spread_bounds_us=[max(0, max(lows)-min(highs)), max(highs)-min(lows)],
                   scope="Matched collective CPU submission spread with measured anchor bounds; not GPU arrival skew or transport duration")
    return {"collectives": result, "clock_policy": "Never subtract unaligned rank trace timestamps; CPU-only alignment bounds"}


def capture(adapter, state, operation, synchronize, context):
    import torch
    from kernel_optimizer.project.communication import bucket_inventory
    output = Path(context["output"])
    options = context["parameters"].get("collector_options", {})
    repetitions = int(options.get("repeats", 1))
    if repetitions < 1:
        raise ValueError("timeline repeats must be positive")
    hook_gradients = bool(options.get("gradient_readiness", True)) and context["world_size"] > 1
    gpu_markers = bool(options.get("gpu_gradient_markers", False)) and hook_gradients and torch.cuda.is_available()
    labels, parameters, handles, missing = {}, {}, [], []
    marker_records, origin_events = [], {}
    if hook_gradients:
        for root, model in adapter.models(state).items():
            for name, parameter in model.named_parameters(remove_duplicate=False):
                entry = parameters.setdefault(id(parameter), (parameter, []))
                entry[1].append(root+"."+name)
        for parameter, names in parameters.values():
            if not parameter.requires_grad:
                continue
            label = "opop.grad_ready::"+hashlib.sha256(json.dumps(names).encode()).hexdigest()[:24]
            labels[label] = names
            register = getattr(parameter, "register_post_accumulate_grad_hook", None)
            if register is None:
                missing.append({"parameters": names, "reason": "post-accumulate API unavailable"})
                continue
            def ready(parameter, label=label):
                with torch.profiler.record_function(label):
                    if gpu_markers and parameter.device.type == "cuda":
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(torch.cuda.current_stream(parameter.device))
                        marker_records.append((label, str(parameter.device), event))
            handles.append(register(ready))
    boot = Path("/proc/sys/kernel/random/boot_id")
    domain = {"host": platform.node(), "boot_id": boot.read_text().strip() if boot.exists() else None,
              "clock": "CLOCK_MONOTONIC"}
    reports = []
    try:
        for index in range(repetitions):
            adapter.reset(state)
            if callable(getattr(adapter, "before_run", None)):
                adapter.before_run(state)
            elif torch.distributed.is_initialized():
                torch.distributed.barrier()
            synchronize()
            marker_records.clear()
            origin_events.clear()
            if gpu_markers:
                for parameter, _ in parameters.values():
                    if parameter.device.type == "cuda" and str(parameter.device) not in origin_events:
                        origin = torch.cuda.Event(enable_timing=True)
                        origin.record(torch.cuda.current_stream(parameter.device))
                        origin_events[str(parameter.device)] = origin
            anchors = []
            def anchor(suffix):
                name = f"opop.clock::{index}:{suffix}"
                before = time.monotonic_ns()
                with torch.profiler.record_function(name):
                    pass
                anchors.append({"name": name, "before_ns": before, "after_ns": time.monotonic_ns()})
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                # Rank-local profiler startup can otherwise masquerade as
                # workload arrival skew. Preparation remains outside the marked
                # workload and outside its elapsed time; no per-op sync is added.
                if callable(getattr(adapter, "before_run", None)):
                    adapter.before_run(state)
                elif torch.distributed.is_initialized():
                    torch.distributed.barrier()
                synchronize()
                anchor("before")
                started = time.perf_counter_ns()
                with torch.profiler.record_function("opop.timeline_workload"):
                    operation()
                    synchronize()
                elapsed = (time.perf_counter_ns()-started)/1e6
                anchor("after")
            path = output/f"timeline-rank-{context['rank']}-{index}.json"
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            inventory = bucket_inventory(adapter, state, {"calls": []}) if hook_gradients else None
            report = analyze(trace, rank=context["rank"], anchors=anchors, domain=domain,
                             gradients=labels, inventory=inventory)
            report["gradient_stream_markers"] = {
                "status": "observed" if marker_records else "not_collected",
                "events": [{"parameters": labels[label], "device": device,
                            "elapsed_from_origin_ms": origin_events[device].elapsed_time(event)}
                           for label, device, event in marker_records],
                "scope": "Optional CUDA event recorded on each gradient callback's current stream; time relative to a rank-local GPU origin. No per-parameter synchronize. Adds event-record overhead; does not prove reducer bucket readiness, unique consumption, or cross-rank GPU clock alignment."}
            report.update(trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          profiled_wall_ms=elapsed, gradient_hook_missing=missing,
                          gradient_hooks_enabled=hook_gradients,
                          clock_anchors=anchors)
            report_path = output/f"timeline-analysis-rank-{context['rank']}-{index}.json"
            write_json(report_path, report)
            reports.append(str(report_path))
    finally:
        for handle in handles:
            handle.remove()
    return {"valid": True, "reports": reports, "collector": "execution_timeline",
            "scope": "Separate passive diagnostic; profiler/gradient hooks perturb execution. Compare against clean evaluation."}
