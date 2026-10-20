"""Joint distributed replacement for the DDP gradient-bucket allreduce_ sites.

Before each guarded c10d::allreduce_ dispatch, both ranks meet at a CPU-side
gloo rendezvous (reusing the adapter's existing out-of-band control group)
and then call the original NCCL op unchanged, so matching collectives are
submitted almost simultaneously instead of with the observed per-collective
skew. The rendezvous enqueues no CUDA kernel; operands, process group,
ReduceOp and async semantics pass through untouched.
"""
from datetime import timedelta

import torch
import torch.distributed as dist

_BARRIER_TIMEOUT = timedelta(seconds=30)

_SITE_NUMEL = {
    "d0938de87369efd500322679": 7087872,
    "b8ade439d7b8bd5fb5c1029b": 39385344,
}


def _guarded_payload(args, kwargs, expected_numel):
    try:
        tensors = args[0] if len(args) >= 1 else kwargs.get("tensors")
        sparse = args[3] if len(args) >= 4 else kwargs.get("sparse_indices")
        if sparse is not None:
            return False
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return False
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.device.type != "cuda" or tensor.dtype != torch.float32:
            return False
        if tensor.dim() != 1 or tensor.stride() != (1,):
            return False
        return tensor.numel() == expected_numel
    except Exception:
        return False


def _make_replacement(call_id, expected_numel, parent, control, world_size, record_route):
    def rendezvous_then_allreduce(*args, **kwargs):
        if not _guarded_payload(args, kwargs, expected_numel):
            record_route(call_id, "fallback:signature-mismatch")
            return parent(*args, **kwargs)
        if world_size <= 1 or control is None or not (dist.is_available() and dist.is_initialized()):
            record_route(call_id, "fallback:no-rendezvous-group")
            return parent(*args, **kwargs)
        try:
            dist.barrier(group=control, timeout=_BARRIER_TIMEOUT)
        except Exception:
            record_route(call_id, "fallback:barrier-error")
            return parent(*args, **kwargs)
        record_route(call_id, "gloo-rendezvous+parent-nccl")
        return parent(*args, **kwargs)

    return rendezvous_then_allreduce


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route") or (lambda *_: None)
    world_size = int(context.get("world_size") or 1)
    control = state.get("control")

    replacements = {}
    for call_id, expected_numel in _SITE_NUMEL.items():
        parent = references.get(call_id)
        if parent is None:
            continue
        replacements[call_id] = _make_replacement(
            call_id, expected_numel, parent, control, world_size, record_route
        )
    return replacements
