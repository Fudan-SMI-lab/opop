"""Joint candidate for the two c10d.allreduce_.default dispatch groups.

Aligns cross-rank CPU submission of every DDP gradient-bucket allreduce with a
CPU-side gloo rendezvous (the adapter's own out-of-band control group) and then
issues the parent NCCL collective unchanged via context.references, so the
pre-existing in-kernel device wait caused by submission skew is no longer
absorbed inside NCCL kernel durations. Evidence, risks, setup work and memory
effects are documented in proposal.json.
"""
from __future__ import annotations

import threading
from datetime import timedelta

import torch
import torch.distributed as dist

_CALL_IDS = (
    "b8ade439d7b8bd5fb5c1029b",
    "d0938de87369efd500322679",
)
_OBSERVED_NUMELS = frozenset({7087872, 39385344})


def _eligible_tensor(value):
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        return None
    tensor = value[0]
    if not isinstance(tensor, torch.Tensor):
        return None
    if (
        tensor.is_cuda
        and tensor.dtype == torch.float32
        and tensor.dim() == 1
        and tensor.stride() == (1,)
        and tensor.numel() in _OBSERVED_NUMELS
    ):
        return tensor
    return None


def _is_gloo_group(group):
    if group is None:
        return False
    if type(group).__name__ == "ProcessGroupGloo":
        return True
    try:
        from torch.distributed import distributed_c10d as _c10d

        return isinstance(group, _c10d.ProcessGroupGloo)
    except Exception:
        return False


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route") or (lambda *_: None)

    parameters = context.get("parameters") or {}
    try:
        stride = int(parameters.get("rendezvous_stride", 1))
    except (TypeError, ValueError):
        stride = 1
    if stride < 1:
        stride = 1

    control = state.get("control")
    lock = threading.Lock()
    rendezvous = {"group": control if _is_gloo_group(control) else None, "resolved": _is_gloo_group(control), "calls": 0}

    def _rendezvous_group():
        if rendezvous["resolved"]:
            return rendezvous["group"]
        if not (dist.is_available() and dist.is_initialized()):
            return None
        with lock:
            if not rendezvous["resolved"]:
                try:
                    created = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
                except Exception:
                    created = None
                if _is_gloo_group(created):
                    rendezvous["group"] = created
                    rendezvous["resolved"] = True
                else:
                    rendezvous["group"] = None
                    rendezvous["resolved"] = True
        return rendezvous["group"]

    def _make_replacement(call_id, parent):
        def replacement(*args, **kwargs):
            tensors = args[0] if args else kwargs.get("tensors")
            if _eligible_tensor(tensors) is None:
                record_route(call_id, "unsupported-signature+parent-nccl-allreduce")
                return parent(*args, **kwargs)
            with lock:
                rendezvous["calls"] += 1
                position = rendezvous["calls"]
            if (position - 1) % stride:
                record_route(call_id, "stride-skipped+parent-nccl-allreduce")
                return parent(*args, **kwargs)
            group = _rendezvous_group()
            if group is None:
                record_route(call_id, "rendezvous-unavailable+parent-nccl-allreduce")
                return parent(*args, **kwargs)
            try:
                dist.barrier(group=group)
            except Exception:
                record_route(call_id, "rendezvous-failed+parent-nccl-allreduce")
                return parent(*args, **kwargs)
            record_route(call_id, "gloo-rendezvous+parent-nccl-allreduce")
            return parent(*args, **kwargs)

        return replacement

    replacements = {}
    for call_id in _CALL_IDS:
        parent = references.get(call_id)
        if parent is not None:
            replacements[call_id] = _make_replacement(call_id, parent)
    return replacements
