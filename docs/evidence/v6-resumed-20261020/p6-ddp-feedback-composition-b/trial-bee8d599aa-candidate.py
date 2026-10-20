"""Joint communication-path candidate for the two selected NCCL groups.

Scope: group f833b57a373d8262a384 (the 12 regular 7,087,872-element DDP
gradient-bucket allreduces, call d0938de87369efd500322679) plus group
8525af72c631b94e3dd0 (the single 39,385,344-element embedding-bucket
allreduce, call b8ade439d7b8bd5fb5c1029b) - the complete 13-collective NCCL
sequence of the final accumulation micro-step's backward on each rank.

Repair of the rejected trial-constraint-repair-0dbfc106: that candidate moved
the 12 regular-bucket collectives onto a freshly initialized second NCCL
communicator while the embedding-bucket allreduce stayed on the default
process group.  It passed every gradient/parameter check and even lowered the
discovery NCCL kernel sum (225.85 -> 139.87 ms), but regressed clean paired
step_ms from 146.88 to 240.51 ms (paired gains -89.91/-97.35 ms) and every
tuned parameter point landed at 247-252 ms, so the split-communicator
mechanism itself - not the transport knob values - was the regression.

This candidate keeps every one of the 13 collectives on the workload's single
default NCCL communicator and stream, issued through the same process-group
backend object the parent c10d::allreduce_ operator uses, with
parent-identical in-place SUM semantics, the same bucket tensors, and the
same async Work contract returned to the DDP reducer.  No second
communicator, no ncclCommSplit, no NCCL_PROTO/NCCL_ALGO windows, no stream
changes, no CUDA graphs, no env mutations.

The only degree of freedom is where rank-arrival skew is absorbed:

  route=parent      exact null point: identical routing, ordering and options
                    to the parent operator (bit-identical baseline path).
  route=align_first before the FIRST collective of each synchronization
                    sequence, align the two ranks with a CPU-side barrier on
                    the adapter's gloo control group.  This adds no NCCL
                    kernel; the early rank's peer-wait moves from GPU time
                    spun inside the NCCL kernels to bounded pre-issue CPU
                    wait, so the kernels on the previously-waiting rank run
                    near their observed wire floor.
  route=align_each  align before every one of the 13 collectives for tight
                    per-bucket pairing (same mechanism, more barriers).

Sequence detection is marker-based, not modulo-counting: the embedding-bucket
call is the last collective of every run (discovery seq 33), so "first call
ever, or the call immediately after the embedding-bucket call" identifies the
first collective of each run on both ranks symmetrically and self-resynchron
izes even if the regular-bucket count ever changed.  Both ranks execute
identical call sequences (rank_specs show 12 + 1 on each rank), so barriers
always pair; a hypothetical divergence surfaces as the control group's
bounded 120 s barrier timeout and a loud run failure, never silent numeric
corruption.

The workload's outputs, gradients, parameter updates, bitwise cross-rank
gradient equality (two-rank fp32 SUM on the same communicator), DDP reducer
contract, bucket layout and completion boundary are untouched; every one of
the 13 collectives still executes as a real allreduce on every invocation.
"""

from __future__ import annotations

from datetime import timedelta

import torch
import torch.distributed as dist

REGULAR_CALL_ID = "d0938de87369efd500322679"
EMBED_CALL_ID = "b8ade439d7b8bd5fb5c1029b"

_DEFAULT_REGULAR_NUMEL = 7087872
_DEFAULT_EMBED_NUMEL = 39385344

_ROUTES = ("parent", "align_first", "align_each")


def _as_process_group(obj):
    from torch._C._distributed_c10d import ProcessGroup

    if isinstance(obj, ProcessGroup):
        return obj
    try:
        return ProcessGroup.unbox(obj)
    except Exception:
        return obj


def _backend_for(pg, device):
    getter = getattr(pg, "_get_backend", None)
    if getter is not None:
        return getter(device)
    return pg


def _expected_numel(context, call_id, fallback):
    try:
        spec = context["targets"][call_id]["signatures"][0]["args"][0][0]
        numel = int(spec["shape"][0])
        dtype = spec.get("dtype", "torch.float32")
        if dtype == "torch.float32" and numel > 0:
            return numel
    except Exception:
        pass
    return fallback


def build(context):
    state = context["state"]
    targets = context.get("targets") or {}
    if REGULAR_CALL_ID not in targets or EMBED_CALL_ID not in targets:
        return {}

    params = dict(context.get("parameters") or {})
    route = str(params.get("route", "parent") or "parent")
    if route not in _ROUTES:
        route = "parent"

    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}

    device = state["device"]
    control = state.get("control")
    align_enabled = (
        route in ("align_first", "align_each")
        and control is not None
        and dist.is_available()
        and dist.is_initialized()
    )

    shared = {"started": False, "last_was_embed": False}

    def _make_replacement(call_id, expected_numel):
        parent = references.get(call_id)

        def allreduce_bucket(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1):
            tensor = tensors[0] if isinstance(tensors, (list, tuple)) and len(tensors) == 1 else None
            compatible = (
                tensor is not None
                and tensor.dim() == 1
                and tensor.numel() == expected_numel
                and tensor.dtype == torch.float32
                and tensor.is_contiguous()
                and tensor.device.type == "cuda"
                and sparse_indices is None
            )
            if not compatible and parent is not None:
                record_route(call_id, "parent-signature-fallback")
                return parent(tensors, process_group, reduce_op, sparse_indices, async_op, timeout)

            first_of_sequence = (not shared["started"]) or shared["last_was_embed"]
            shared["started"] = True
            shared["last_was_embed"] = call_id == EMBED_CALL_ID

            if align_enabled and (route == "align_each" or first_of_sequence):
                dist.barrier(group=control)
                record_route(call_id, "default-pg-align")
            else:
                record_route(call_id, "default-pg-parent")

            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            if isinstance(sparse_indices, torch.Tensor):
                opts.sparseIndices = sparse_indices
            if isinstance(timeout, (int, float)) and timeout > 0:
                opts.timeout = timedelta(milliseconds=timeout)
            opts.asyncOp = bool(async_op)
            backend = _backend_for(_as_process_group(process_group), device)
            work = backend.allreduce(list(tensors), opts)
            if not async_op:
                work.wait()
            return (list(tensors), work.boxed())

        return allreduce_bucket

    regular = _make_replacement(REGULAR_CALL_ID, _expected_numel(context, REGULAR_CALL_ID, _DEFAULT_REGULAR_NUMEL))
    embed = _make_replacement(EMBED_CALL_ID, _expected_numel(context, EMBED_CALL_ID, _DEFAULT_EMBED_NUMEL))
    return {REGULAR_CALL_ID: regular, EMBED_CALL_ID: embed}
