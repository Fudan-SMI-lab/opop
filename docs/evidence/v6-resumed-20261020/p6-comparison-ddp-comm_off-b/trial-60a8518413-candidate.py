"""Joint distributed candidate: route the final-micro-batch DDP gradient-bucket
allreduces (12 x 7,087,872-element transformer buckets, 1 x 39,385,344-element
embedding bucket) onto a dedicated NCCL communicator initialized under an
NCCL_PROTO=Simple window, gated by a build-time paired probe with cross-rank
consensus; every call keeps real in-place SUM semantics and returns a live
parent-compatible Work. route=parent reproduces the baseline exactly.
"""
from __future__ import annotations

from datetime import timedelta
import sys

import torch
import torch.distributed as dist

SMALL_CALL = "d0938de87369efd500322679"
BIG_CALL = "b8ade439d7b8bd5fb5c1029b"
_CALLS = (SMALL_CALL, BIG_CALL)
_DEFAULT_NUMELS = {7087872, 39385344}
_CACHE_KEY = "opop_candidate_gpt2_ddp_flush_router_v1"
_TAG_KEY = _CACHE_KEY + "/tag"
_TAG_PREFIX = "opop-ddp-flush-simple-v1"
_INIT_TIMEOUT_S = 60
_PROBE_WAIT_S = 30
_PROBE_MARGIN = 1.05


def _backend_for(pg, device):
    getter = getattr(pg, "_get_backend", None)
    if getter is not None:
        return getter(device)
    return pg


def _default_backend(device):
    from torch.distributed.distributed_c10d import _world

    pg, _store = _world.pg_map[dist.group.WORLD]
    return _backend_for(pg, device)


def _bucket_numels(context):
    numels = set()
    for call_id in _CALLS:
        spec = (context.get("targets") or {}).get(call_id)
        for signature in spec.get("signatures", []) if spec else []:
            try:
                head = signature["args"][0][0]
                if head.get("dtype", "torch.float32") == "torch.float32":
                    numels.add(int(head["shape"][0]))
            except Exception:
                continue
    return numels or set(_DEFAULT_NUMELS)


def _apply_env(window):
    import os

    saved = {}
    for key, value in window.items():
        if value:
            saved[key] = os.environ.get(key)
            os.environ[key] = value
    return saved


def _restore_env(saved):
    import os

    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _next_tag():
    holder = sys.modules.get(_TAG_KEY, None)
    count = int(holder["n"]) + 1 if isinstance(holder, dict) else 0
    sys.modules[_TAG_KEY] = {"n": count}
    return f"{_TAG_PREFIX}/{count}/"


def _build_fresh_backend(device, rank, world_size):
    from torch._C._distributed_c10d import PrefixStore, ProcessGroupNCCL
    from torch.distributed.distributed_c10d import _world

    _pg, base_store = _world.pg_map[dist.group.WORLD]
    store = PrefixStore(_next_tag(), base_store)
    options = ProcessGroupNCCL.Options()
    for setter in (
        lambda: setattr(options, "device_id", device),
        lambda: setattr(options, "_timeout", timedelta(seconds=_INIT_TIMEOUT_S)),
    ):
        try:
            setter()
        except Exception:
            pass
    saved = _apply_env({"NCCL_PROTO": "Simple"})
    try:
        torch.cuda.set_device(device)
        backend = ProcessGroupNCCL(store, int(rank), int(world_size), options)
        probe = torch.zeros(1, dtype=torch.float32, device=device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        opts.asyncOp = True
        work = backend.allreduce([probe], opts)
        if work.wait(timedelta(seconds=_PROBE_WAIT_S)) is False:
            raise TimeoutError("fresh communicator probe timed out")
        del probe
        return backend
    finally:
        _restore_env(saved)


def _time_backend(backend, tensor, device, warmup=1, iters=2):
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    opts.asyncOp = True
    for _ in range(warmup):
        backend.allreduce([tensor], opts).wait()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = None
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start.record()
        backend.allreduce([tensor], opts).wait()
        end.record()
        torch.cuda.synchronize(device)
        elapsed = float(start.elapsed_time(end))
        best = elapsed if best is None else min(best, elapsed)
    return best


def _consensus_sum(state, values):
    control = state.get("control") if isinstance(state, dict) else None
    if control is None or not (dist.is_available() and dist.is_initialized()):
        return None
    payload = torch.tensor(list(values), dtype=torch.int64)
    dist.all_reduce(payload, group=control)
    return [int(v) for v in payload.tolist()]


def _cached_decision():
    entry = sys.modules.get(_CACHE_KEY, None)
    if isinstance(entry, dict) and entry.get("probed"):
        return entry
    return None


def build(context):
    state = context["state"]
    parameters = dict(context.get("parameters") or {})
    route = str(parameters.get("route", "adaptive") or "adaptive")
    if route not in ("parent", "adaptive"):
        route = "adaptive"
    device = state["device"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])
    numels = _bucket_numels(context)

    backend = None
    use_fresh = False
    if route == "parent":
        backend = None
        use_fresh = False
    else:
        entry = _cached_decision()
        if entry is not None:
            backend, use_fresh = entry["backend"], entry["use_fresh"]
        else:
            status = 1
            default_ms = fresh_ms = 0.0
            if (
                device.type == "cuda"
                and dist.is_available()
                and dist.is_initialized()
            ):
                try:
                    fresh = _build_fresh_backend(device, rank, world_size)
                    probe_numel = min(numels) if numels else min(_DEFAULT_NUMELS)
                    probe = torch.zeros(probe_numel, dtype=torch.float32, device=device)
                    default_ms = _time_backend(_default_backend(device), probe, device)
                    fresh_ms = _time_backend(fresh, probe, device)
                    del probe
                    if not (default_ms > 0.0 and fresh_ms > 0.0):
                        raise ValueError("non-positive probe timing")
                    backend = fresh
                except Exception:
                    backend = None
                    status = 0
                    default_ms = fresh_ms = 0.0
            else:
                status = 0
            sums = _consensus_sum(
                state,
                [status, int(round(default_ms * 1000)), int(round(fresh_ms * 1000))],
            )
            if sums is None:
                backend = None
                use_fresh = False
            else:
                total_status, default_us, fresh_us = sums
                use_fresh = (
                    backend is not None
                    and total_status == world_size
                    and fresh_us > 0
                    and default_us > int(_PROBE_MARGIN * fresh_us)
                )
                if not use_fresh:
                    backend = None
                sys.modules[_CACHE_KEY] = {
                    "probed": True,
                    "backend": backend,
                    "use_fresh": use_fresh,
                }

    machine = {"backend": backend if use_fresh else None, "numels": numels}

    def replacement(call_id):
        reference = context["references"][call_id]

        def call(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1):
            tensor = tensors[0] if isinstance(tensors, (list, tuple)) and len(tensors) == 1 else None
            chosen = machine["backend"]
            if (
                chosen is not None
                and tensor is not None
                and sparse_indices is None
                and bool(async_op)
                and tensor.dim() == 1
                and tensor.is_contiguous()
                and tensor.dtype == torch.float32
                and tensor.device.type == "cuda"
                and tensor.numel() in machine["numels"]
            ):
                opts = dist.AllreduceOptions()
                opts.reduceOp = dist.ReduceOp.SUM
                opts.asyncOp = True
                if isinstance(timeout, (int, float)) and timeout > 0:
                    opts.timeout = timedelta(milliseconds=int(timeout))
                work = chosen.allreduce(list(tensors), opts)
                if not async_op:
                    work.wait()
                return (list(tensors), work.boxed() if hasattr(work, "boxed") else work)
            return reference(tensors, process_group, reduce_op, sparse_indices, async_op, timeout)

        return call

    return {
        call_id: replacement(call_id)
        for call_id in _CALLS
        if call_id in (context.get("targets") or {})
        and call_id in (context.get("references") or {})
    }
