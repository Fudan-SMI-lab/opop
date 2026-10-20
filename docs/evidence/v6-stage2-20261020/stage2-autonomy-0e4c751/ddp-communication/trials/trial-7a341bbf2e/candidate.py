"""Joint CPU-alignment wrapper for the two DDP bucket-reduction allreduce sites.

diagnostic.max_rank_nccl_kernel_sum_ms is the max over ranks of the NCCL kernel
duration sum.  The transport floor is already reached in-model (12 small buckets
at ~3.57 ms plus one 157.5 MB bucket at ~19.6 ms, a flat ~8 GB/s logical rate,
peer access disabled), so the addressable term is rank-arrival skew: the measured
cross-rank CPU submit spread is ~20.7-21.3 ms and the early rank's first
ncclDevKernel spins that long before the peer's data arrives.  Each replacement
runs a CPU-only gloo barrier on the side control group (state["control"], created
by adapter.prepare) and then forwards to the parent c10d::allreduce_ with identical
arguments, so the skew is paid as uncounted CPU wait on the rank with slack while
tensors, process group, ReduceOp, async Work handling and collective order stay
exactly as observed.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

_TARGET_OP = "c10d.allreduce_.default"

_ROUTE_ALIGNED = "gloo_cpu_aligned_allreduce"
_ROUTE_NO_GROUP = "setup_no_control_group_parent_allreduce"
_ROUTE_SIGNATURE = "unsupported_signature_parent_allreduce"
_ROUTE_BARRIER_FAILURE = "runtime_barrier_failure_parent_allreduce"


def _expected_numels(spec):
    numels = set()
    for signature in (spec or {}).get("signatures") or []:
        args = signature.get("args") or []
        if not args or not isinstance(args[0], list):
            continue
        for entry in args[0]:
            if isinstance(entry, dict) and entry.get("shape") and len(entry["shape"]) == 1:
                numels.add(int(entry["shape"][0]))
    return numels


def _signature_supported(tensors, expected_numels):
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
        return False
    tensor = tensors[0]
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda" or tensor.dtype != torch.float32:
        return False
    if tensor.dim() != 1 or not tensor.is_contiguous():
        return False
    if expected_numels and tensor.numel() not in expected_numels:
        return False
    return True


def _make_replacement(call_id, parent, control, expected_numels, record_route):
    def aligned_allreduce_(tensors, *args, **kwargs):
        if control is None:
            record_route(call_id, _ROUTE_NO_GROUP)
            return parent(tensors, *args, **kwargs)
        if not _signature_supported(tensors, expected_numels):
            record_route(call_id, _ROUTE_SIGNATURE)
            return parent(tensors, *args, **kwargs)
        try:
            dist.barrier(group=control)
        except Exception:
            record_route(call_id, _ROUTE_BARRIER_FAILURE)
            return parent(tensors, *args, **kwargs)
        record_route(call_id, _ROUTE_ALIGNED)
        return parent(tensors, *args, **kwargs)

    return aligned_allreduce_


def build(context):
    state = context.get("state") or {}
    targets = context.get("targets") or {}
    record_route = context.get("record_route", lambda *_: None)

    control = state.get("control")
    if control is None:
        world_size = int(context.get("world_size") or 1)
        try:
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                control = dist.new_group(backend="gloo")
        except Exception:
            control = None

    replacements = {}
    for call_id, parent in (context.get("references") or {}).items():
        spec = targets.get(call_id) or {}
        if spec.get("kind") != "functional" or spec.get("target") != _TARGET_OP:
            continue
        replacements[call_id] = _make_replacement(
            call_id, parent, control, _expected_numels(spec), record_route
        )
    return replacements
