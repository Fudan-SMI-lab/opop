"""Observed-size collective reference experiments, never project objective samples."""
import json
import statistics
import time


def layouts(discovery, world_size):
    selected, omitted = {}, []
    ranks = discovery.get("communication", {}).get("ranks", [])
    for row in ranks[0].get("collectives", []) if ranks else []:
        try:
            participants = json.loads(row["participants"]) if isinstance(row["participants"], str) else row["participants"]
        except (ValueError, TypeError):
            participants = None
        supported = (row["operation"] == "allreduce" and participants == list(range(world_size))
            and row.get("payload_status") == "observed" and row.get("input_elements", 0) > 0
            and any("_Sum_" in k["name"] for k in row.get("kernels", [])))
        if not supported:
            omitted.append({"operation": row["operation"], "sequence": row.get("sequence"),
                            "reason": "only observed full-world SUM allreduce layouts are supported by this collector"})
            continue
        key = (row["dtype"], row["input_elements"])
        selected.setdefault(key, {"operation": "allreduce", "reduction": "sum", "dtype": key[0],
            "elements": key[1], "logical_bytes": row["logical_input_bytes"], "source_collectives": []})["source_collectives"].append(
                {"group": row["process_group"], "sequence": row["sequence"], "calls": row["calls"]})
    return {"layouts": list(selected.values()), "omitted": omitted,
            "scope": "synthetic contiguous reference at observed payload size/dtype and rank count; not a reproduction of model scheduling or a lower bound"}


def capture(context, options):
    import torch
    import torch.distributed as dist
    if context["world_size"] < 2 or not dist.is_initialized():
        return {"valid": True, "calibration": {"status": "not_applicable", "reason": "requires a distributed task"}}
    group = dist.new_group(backend="gloo")
    aliases = {"Float": torch.float32, "Double": torch.float64, "Half": torch.float16,
               "BFloat16": torch.bfloat16, "Long": torch.int64, "Int": torch.int32}
    warmup, repeats = int(options.get("warmup", 2)), int(options.get("repeats", 5))
    if warmup < 0 or repeats < 1:
        raise ValueError("calibration warmup/repeats are invalid")
    rows = []
    try:
        for layout in options["layouts"]:
            dtype = aliases.get(layout["dtype"])
            if dtype is None:
                rows.append({**layout, "status": "unavailable", "reason": "unsupported observed dtype"})
                continue
            tensor, failure = None, None
            try:
                tensor = torch.empty(layout["elements"], dtype=dtype, device=context["device"])
            except torch.OutOfMemoryError as exc:
                failure = str(exc)
            ready = torch.tensor(int(tensor is not None))
            dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
            if not ready.item():
                rows.append({**layout, "status": "unavailable", "reason": failure or "another rank could not allocate the observed payload"})
                del tensor
                continue
            samples, passed = [], True
            expected = context["world_size"]*(context["world_size"]+1)//2
            for index in range(warmup+repeats):
                tensor.fill_(context["rank"]+1)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                started = time.perf_counter_ns()
                work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
                work.wait()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns()-started)/1e6
                # Check all elements without an additional full-size comparison tensor.
                passed = passed and tensor.amin().item() == expected and tensor.amax().item() == expected
                if index >= warmup:
                    samples.append(elapsed)
            quality = torch.tensor(int(passed)); dist.all_reduce(quality, op=dist.ReduceOp.MIN, group=group)
            rows.append({**layout, "status": "observed" if quality.item() else "failed", "quality": bool(quality.item()),
                "samples_ms": samples, "median_completion_ms": statistics.median(samples),
                "logical_payload_bytes_per_second": layout["logical_bytes"]/(statistics.median(samples)/1000),
                "completion": "async Work wait followed by device synchronization; control uses separate gloo group outside timer"})
            del tensor
    finally:
        dist.destroy_process_group(group)
    return {"valid": all(row["status"] != "failed" for row in rows), "calibration": {
        "status": "observed" if any(row["status"] == "observed" for row in rows) else "unavailable",
        "rank": context["rank"], "rows": rows, "warmup": warmup, "repeats": repeats,
        "omitted": options.get("omitted", []), "interpretation": options["scope"]+
        "; effective logical payload rate is not busbw or wire throughput; do not infer model critical-path savings"}}
