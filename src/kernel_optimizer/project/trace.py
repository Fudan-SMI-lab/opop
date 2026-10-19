"""Join launch correlation and autograd sequence evidence without cross-rank clocks."""
from collections import defaultdict


def associate(trace, events, calls=None):
    calls = calls or {}
    labels = {"opop::"+str(event["sequence"]): event["call_id"] for event in events}
    timelines = defaultdict(list)
    roots, backward, kernels = [], [], []
    for row in trace.get("traceEvents", []):
        if row.get("ph") != "X":
            continue
        args, name = row.get("args", {}), row.get("name", "")
        key = (row.get("pid"), row.get("tid"))
        if name in labels:
            ident = labels[name]
            timelines[key].extend(((row["ts"], 1, id(row), ((ident,), False)),
                                   (row["ts"]+row.get("dur", 0), 0, id(row), None)))
            if calls.get(ident, {}).get("kind") == "project":
                roots.append((row.get("pid"), row["ts"], row["ts"]+row.get("dur", 0), ident))
        if row.get("cat") in {"cuda_runtime", "cuda_driver"} and "correlation" in args:
            timelines[key].append((row["ts"], 2, args["correlation"], None))
        if name.startswith("autograd::engine::evaluate_function:"):
            backward.append(row)
        elif row.get("cat") == "cpu_op" and "Sequence number" in args and not args.get("Fwd thread id", 0):
            timelines[key].append((row["ts"], 3, args["Sequence number"], None))
        if row.get("cat") == "kernel":
            kernels.append({"name": name, "start_us": row["ts"], "duration_us": row.get("dur", 0),
                "correlation": args.get("correlation"), "external_id": args.get("External id"),
                "stream": args.get("stream"), "device": args.get("device"),
                "launch_resources": {k: args[k] for k in ("registers per thread", "shared memory", "grid", "block", "graph id", "graph node id") if k in args},
                "communication": "Collective name" in args or "nccl" in name.lower()})
    sequences = {}
    ambiguous = set()
    for (pid, tid), timeline in timelines.items():
        active = {}
        for _, kind, token, data in sorted(timeline, key=lambda event: (event[0], event[1])):
            if kind == 0:
                active.pop(token, None)
            elif kind == 1:
                active[token] = data
            elif kind == 3:
                owners = tuple(dict.fromkeys(c for value, _ in active.values() for c in value))
                if not owners:
                    continue
                key = (pid, token)
                if key in sequences and set(sequences[key]) != set(owners):
                    ambiguous.add(key)
                    owners = tuple(c for c in owners if c in sequences[key])
                sequences[key] = owners
    linked_ranges = 0
    for row in backward:
        key = (row.get("pid"), row.get("tid"))
        owners = sequences.get((row.get("pid"), row.get("args", {}).get("Sequence number")), ())
        if owners:
            linked_ranges += 1
        timelines[key].extend(((row["ts"], 1, ("backward", id(row)), (owners, True)),
                               (row["ts"]+row.get("dur", 0), 0, ("backward", id(row)), None)))
    launches = {}
    for (pid, tid), timeline in timelines.items():
        active = {}
        for timestamp, kind, token, data in sorted(timeline, key=lambda event: (event[0], event[1])):
            if kind == 0:
                active.pop(token, None)
            elif kind == 1:
                active[token] = data
            elif kind == 2:
                owners = list(dict.fromkeys(c for value, _ in active.values() for c in value))
                is_backward = any(back for _, back in active.values())
                # The project interval owns its worker threads; narrower ownership
                # still requires an observed annotation or autograd sequence link.
                for root_pid, start, end, ident in roots:
                    if root_pid == pid and start <= timestamp <= end and ident not in owners:
                        owners.insert(0, ident)
                phase = "backward" if is_backward else next((calls[c]["phase"] for c in reversed(owners) if c in calls), "unknown")
                launches[token] = (owners, phase)
    totals = defaultdict(lambda: {"cuda_us": 0.0, "kernels": 0, "communication_us": 0.0, "phase_cuda_us": {}})
    matched, detailed, backward_kernels = 0, 0, 0
    for kernel in kernels:
        owners, phase = launches.get(kernel["correlation"], ([], "unknown"))
        kernel["calls"], kernel["phase"] = owners, phase
        matched += bool(owners)
        detailed += any(calls.get(ident, {}).get("kind") != "project" for ident in owners)
        backward_kernels += phase == "backward" and bool(owners)
        for ident in owners:
            total = totals[ident]
            total["cuda_us"] += kernel["duration_us"]
            total["kernels"] += 1
            total["phase_cuda_us"][phase] = total["phase_cuda_us"].get(phase, 0)+kernel["duration_us"]
            if kernel["communication"]:
                total["communication_us"] += kernel["duration_us"]
    return {"kernels": kernels, "call_resources": dict(totals),
        "coverage": {"cuda_kernels": len(kernels), "associated_kernels": matched,
            "unassociated_kernels": len(kernels)-matched, "associated_below_project": detailed,
            "root_only_kernels": matched-detailed, "backward_associated_kernels": backward_kernels,
            "autograd_ranges": len(backward), "forward_linked_autograd_ranges": linked_ranges,
            "ambiguous_forward_sequences": len(ambiguous)},
        "timing_scope": "Diagnostic CUDA durations. Nested totals overlap. Backward owners use native autograd sequence links; root-only and ambiguous mappings remain explicit."}
