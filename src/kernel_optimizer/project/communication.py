"""Optional communication observations, normalized from actual trace metadata."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from kernel_optimizer.models.project import CapabilityManifest, Observation
from kernel_optimizer.project.transactions import identity


def _command(args, destination):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        destination.write_text(result.stdout+result.stderr)
        return {"status": "observed" if result.returncode == 0 else "unavailable",
                "returncode": result.returncode, "evidence": str(destination)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def probe(context, policy="auto", tool_paths=None):
    import torch
    count = len(context["binding"].get("requested_devices", []))
    count = max(count, context["world_size"], torch.cuda.device_count())
    applicable = policy != "off" and count > 1
    output = Path(context["output"])
    fields, tools = [], {}
    if not applicable:
        reason = "communication diagnostics disabled" if policy == "off" else "single-device task"
        fields.append(Observation(name="communication", status="not_applicable", unit="capability", reason=reason))
    else:
        for left in range(torch.cuda.device_count()):
            for right in range(torch.cuda.device_count()):
                if left != right:
                    fields.append(Observation(name=f"peer_access.{left}.{right}", status="observed", unit="boolean",
                                              value=float(torch.cuda.can_device_access_peer(left, right))))
        for name, args in {
            "topology": ["nvidia-smi", "topo", "-m"], "nvlink": ["nvidia-smi", "nvlink", "-s"],
            "pcie_link": ["nvidia-smi", "--query-gpu=uuid,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max", "--format=csv"]}.items():
            tools[name] = _command(args, output/f"capability-{name}-rank-{context['rank']}.txt")
        for name in ("nsys", "ncu"):
            path = (tool_paths or {}).get(name) or shutil.which(name)
            tools[name] = {"path": path, "status": "available" if path else "unavailable"}
        from kernel_optimizer.project.telemetry import probe as probe_telemetry
        from kernel_optimizer.runtime.execution import write_json
        device_observations = probe_telemetry(context["binding"]["requested_uuid"])
        telemetry_path = output/f"capability-telemetry-rank-{context['rank']}.json"
        write_json(telemetry_path, device_observations)
        tools["device_telemetry"] = {**device_observations, "evidence": str(telemetry_path)}
        for reading in device_observations.get("fields", []):
            measured = reading["status"] == "observed"
            fields.append(Observation(name=reading["name"], status="observed" if measured else "unavailable",
                value=reading.get("value") if measured else None, unit=reading["unit"], entity=reading["entity"],
                reason=None if measured else reading.get("reason", reading["status"]), evidence=(str(telemetry_path),)))
        for name in ("pcie_bytes", "nvlink_bytes", "hardware_utilization"):
            fields.append(Observation(name=name, status="unknown", unit="bytes" if name.endswith("bytes") else "fraction",
                reason="Hardware counters require a successful collector-specific probe; topology and logical payload do not measure traffic"))
    return CapabilityManifest(id=identity({"rank": context["rank"], "policy": policy, "tools": tools}),
        supported_scopes=("group", "joint", "region", "execution_plan"), fields=tuple(fields), tools=tools)


def intervals(rows):
    merged = []
    for start, end in sorted(rows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def overlap(left, right):
    a, b, total = 0, 0, 0.0
    while a < len(left) and b < len(right):
        total += max(0, min(left[a][1], right[b][1])-max(left[a][0], right[b][0]))
        if left[a][1] < right[b][1]:
            a += 1
        else:
            b += 1
    return total


_DTYPE_BYTES = {"float": 4, "float32": 4, "half": 2, "float16": 2, "bfloat16": 2,
                "double": 8, "float64": 8, "long": 8, "int64": 8, "int": 4, "int32": 4,
                "short": 2, "int16": 2, "char": 1, "byte": 1, "int8": 1, "uint8": 1, "bool": 1}


def trace_window(trace, label):
    """Keep operations submitted inside one explicit, completion-bounded workload."""
    windows = [r for r in trace.get("traceEvents", []) if r.get("ph") == "X" and r.get("name") == label]
    if len(windows) != 1:
        return trace, None
    start, end = windows[0]["ts"], windows[0]["ts"]+windows[0]["dur"]
    window = {"start_us": start, "end_us": end,
              "scope": "explicit workload including final completion synchronization; excludes profiler preparation"}
    return {**trace, "traceEvents": [r for r in trace["traceEvents"]
            if "ts" in r and start <= r["ts"] < end]}, window


def normalize(trace, associations, rank):
    kernels = [row for row in trace.get("traceEvents", []) if row.get("cat") == "kernel" and row.get("ph") == "X"]
    joined = {k["correlation"]: k for k in associations["kernels"]}
    collectives, unmatched = {}, []
    communication, compute = [], []
    communication_duration = 0.0
    for row in kernels:
        args = row.get("args", {})
        is_comm = "Collective name" in args or "nccl" in row.get("name", "").lower()
        (communication if is_comm else compute).append((row["ts"], row["ts"]+row.get("dur", 0)))
        if not is_comm:
            continue
        communication_duration += row.get("dur", 0)
        required = ("Seq", "Process Group Name", "Collective name")
        if any(field not in args for field in required):
            unmatched.append({"kernel": row["name"], "correlation": args.get("correlation"), "reason": "collective identity metadata unavailable"})
            continue
        key = (str(args["Process Group Name"]), args["Seq"], args["Collective name"])
        item = collectives.setdefault(key, {"rank": rank, "process_group": key[0], "sequence": key[1], "operation": key[2],
            "participants": args.get("Process Group Ranks"), "dtype": args.get("dtype"),
            "input_elements": args.get("In msg nelems"), "output_elements": args.get("Out msg nelems"),
            "input_splits": args.get("In split size"), "output_splits": args.get("Out split size"),
            "kernels": [], "calls": [], "parameter_names": [], "parameter_calls": [],
            "logical_input_bytes": None, "payload_status": "unknown"})
        width = _DTYPE_BYTES.get(str(item["dtype"]).lower())
        if width is not None and item["input_elements"] is not None:
            item["logical_input_bytes"] = int(item["input_elements"])*width
            item["payload_status"] = "observed"
        association = joined.get(args.get("correlation"), {})
        item["calls"] = sorted(set(item["calls"])|set(association.get("calls", [])))
        for ownership in association.get("payload_ownership", []):
            item["parameter_names"] = sorted(set(item["parameter_names"])|set(ownership["parameters"]))
            item["parameter_calls"] = sorted(set(item["parameter_calls"])|set(ownership["calls"]))
        item["kernels"].append({"name": row["name"], "start_us": row["ts"], "duration_us": row.get("dur", 0),
                                "stream": args.get("stream"), "correlation": args.get("correlation")})
    comm_union, compute_union = intervals(communication), intervals(compute)
    union_us = sum(end-start for start, end in comm_union)
    overlap_us = overlap(comm_union, compute_union)
    return {"rank": rank, "status": "observed" if communication else "not_observed",
        "collectives": list(collectives.values()), "unmatched_communication_kernels": unmatched,
        "communication_kernel_count": len(communication), "communication_kernel_sum_ms": communication_duration/1000,
        "communication_union_ms": union_us/1000, "compute_overlap_ms": overlap_us/1000,
        "communication_without_compute_ms": (union_us-overlap_us)/1000,
        "logical_input_bytes": sum(item["logical_input_bytes"] for item in collectives.values())
            if not unmatched and all(item["payload_status"] == "observed" for item in collectives.values()) else None,
        "payload_coverage": {"observed_collectives": sum(item["payload_status"] == "observed" for item in collectives.values()),
                             "total_collectives": len(collectives)},
        "interpretation": "Logical per-rank payload is counted once per collective, not per kernel. Kernel time includes waiting. Interval overlap is local to one trace clock; uncovered communication is not a measured critical path or causal speedup."}


def match_ranks(observations):
    groups = {}
    for observation in observations:
        for item in observation["collectives"]:
            key = (item["process_group"], item["sequence"], item["operation"], item["participants"])
            groups.setdefault(key, []).append(item)
    matched, incomplete = [], []
    for key, items in groups.items():
        try:
            participants = set(json.loads(key[3]))
        except (TypeError, ValueError):
            participants = set()
        record = {"process_group": key[0], "sequence": key[1], "operation": key[2],
                  "ranks": sorted(item["rank"] for item in items), "participants": sorted(participants),
                  "payload_by_rank": {str(item["rank"]): item["logical_input_bytes"] for item in items},
                  "calls_by_rank": {str(item["rank"]): item["calls"] for item in items}}
        record["parameter_calls_by_rank"] = {str(item["rank"]): item.get("parameter_calls", []) for item in items}
        complete = participants and {item["rank"] for item in items} == participants and len(items) == len(participants)
        (matched if complete else incomplete).append(record)
    return {"matched": matched, "incomplete": incomplete, "clock_policy": "No subtraction of timestamps from different rank clocks"}


def transports(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            if " via " in line and ("Channel " in line or " -> " in line):
                rows.append({"transport": line.split(" via ", 1)[1].strip(), "evidence": str(path), "line": line})
    return {"status": "observed" if rows else "unknown", "channels": rows,
            "reason": None if rows else "NCCL transport selection was not present in the captured initialization logs"}


def available_fields(observation):
    available = set()
    if observation.get("communication_kernel_count", 0):
        available.add("nccl_kernel_duration")
    if observation.get("collectives") and not observation.get("unmatched_communication_kernels"):
        available.add("collective_identity")
        if observation.get("logical_input_bytes") is not None:
            available.add("logical_payload")
    if observation.get("transport", {}).get("status") == "observed":
        available.add("transport")
    return available


def bucket_inventory(adapter, state, discovered):
    """Passive post-run reducer inventory; never replace an application's comm hook."""
    import torch
    roots = adapter.models(state)
    parameter_names, seen, buckets, missing = {}, set(), [], []
    for root, model in roots.items():
        for name, parameter in model.named_parameters(remove_duplicate=False):
            parameter_names.setdefault(id(parameter), []).append(root+("."+name if name else ""))
    module_calls = [call for call in discovered["calls"] if call["kind"] == "module"]
    inventory_roots = dict(roots)
    if isinstance(state, dict):
        inventory_roots.update({"state:"+name: value for name, value in state.items() if isinstance(value, torch.nn.Module)})
    for root, model in inventory_roots.items():
        for path, module in model.named_modules():
            if not isinstance(module, torch.nn.parallel.DistributedDataParallel) or id(module) in seen:
                continue
            seen.add(id(module))
            name = root+("."+path if path else "")
            getter = getattr(module.reducer, "_get_zeros_like_grad_buckets", None)
            if getter is None:
                missing.append({"module": name, "reason": "installed reducer exposes no passive bucket inventory"})
                continue
            try:
                actual = getter()
                for bucket in actual:
                    names = sorted({n for p in bucket.parameters() for n in parameter_names.get(id(p), ())})
                    direct_modules = {p.rpartition(".")[0] for p in names}
                    calls = sorted(c["id"] for c in module_calls if c["target"] in direct_modules)
                    tensor = bucket.buffer()
                    buckets.append({"module": name, "bucket_index": bucket.index(), "is_last": bucket.is_last(),
                        "parameters": names, "calls": calls, "elements": tensor.numel(),
                        "dtype": str(tensor.dtype), "logical_bytes": tensor.numel()*tensor.element_size()})
                del actual
            except (RuntimeError, AttributeError) as exc:
                missing.append({"module": name, "reason": f"{type(exc).__name__}: {exc}"})
    return {"status": "observed" if buckets else "unavailable" if seen else "not_applicable", "buckets": buckets,
        "missing": missing, "method": "installed reducer passive bucket/parameter inventory after warmed profiled run",
        "interpretation": "Direct parameter ownership is many-to-many. No comm hook is installed or replaced. This inventory alone does not identify a timed collective or allocate its cost among member groups."}


def attach_bucket_candidates(communication, inventory):
    communication["bucket_inventory"] = inventory
    for collective in communication.get("collectives", []):
        candidates = [bucket for bucket in inventory["buckets"]
                      if bucket["logical_bytes"] == collective["logical_input_bytes"]
                      and collective["operation"].lower().replace("_", "") == "allreduce"]
        collective["bucket_candidates"] = [{"module": b["module"], "bucket_index": b["bucket_index"], "calls": b["calls"]} for b in candidates]
        owners = set(collective.get("parameter_names", []))
        exact = [b for b in candidates if owners and owners == set(b["parameters"])]
        collective["observed_buckets"] = [{"module": b["module"], "bucket_index": b["bucket_index"], "calls": b["calls"]} for b in exact]
        collective["bucket_attribution"] = ("observed gradient-storage membership plus matched parameter inventory" if exact else
            "payload-compatible candidates only; event-to-bucket identity remains unknown")


def sample_metric(adapter, state, operation, synchronize, context, source, repeats):
    """Plain profiler samples: no Python discovery hooks in the scored diagnostic."""
    import torch
    import time
    values, evidence, perturbation = [], [], []
    if source["collector"] != "torch_communication":
        raise ValueError(f"unsupported metric collector: {source['collector']}")
    for index in range(repeats):
        adapter.reset(state)
        if callable(getattr(adapter, "before_run", None)):
            adapter.before_run(state)
        elif torch.distributed.is_initialized():
            torch.distributed.barrier()
        synchronize()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            # Profiler initialization is rank-local and can introduce arrival
            # skew. Prepare again outside the explicitly marked score window.
            if callable(getattr(adapter, "before_run", None)):
                adapter.before_run(state)
            elif torch.distributed.is_initialized():
                torch.distributed.barrier()
            synchronize()
            started = time.perf_counter_ns()
            with torch.profiler.record_function("opop.communication_metric_workload"):
                operation(); synchronize()
            perturbation.append((time.perf_counter_ns()-started)/1e6)
        path = Path(context["output"])/f"metric-rank-{context['rank']}-{index}.json"
        profiler.export_chrome_trace(str(path))
        trace, window = trace_window(json.loads(path.read_text()), "opop.communication_metric_workload")
        if window is None:
            raise ValueError("communication metric workload boundary is missing or ambiguous")
        observed = normalize(trace, {"kernels": []}, context["rank"])
        if observed["status"] != "observed":
            raise ValueError("required communication metric was not observed")
        value = observed.get(source["field"])
        if not isinstance(value, (float, int)):
            raise ValueError(f"required collector field is unavailable: {source['field']}")
        values.append(float(value)); evidence.append(str(path))
    return {"values": values, "evidence": evidence, "profiled_wall_ms": perturbation,
            "protocol": "profiler_start_then_rank_preparation_v2; explicit completion-bounded workload, excluding preparation collectives",
            "scope": "separate plain PyTorch diagnostic; kernel time includes waiting, not wire or critical-path time"}
