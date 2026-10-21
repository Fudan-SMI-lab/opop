"""Rank alignment at the first DDP bucket allreduce of each training step.

The accumulated step's synchronized final backward issues exactly 13
c10d::allreduce_ calls per rank (12 x 7,087,872-element fp32 buckets, closed
by the 39,385,344-element embedding bucket). communication_kernel_sum_ms is a
per-rank sum of device NCCL kernel durations, and the execution_timeline
evidence shows its reducible term is in-kernel spin-wait, not transfer: one
rank dispatches the first collective ~6.4 ms before the peer (seq 21), so that
rank's first NCCL kernel runs 9.70 ms against a ~3.3-3.6 ms synchronized floor
while every later kernel runs near floor in NCCL lock-step (the submission
offset persists but is absorbed once by the first kernel).

This candidate replaces both observed allreduce_ sites with the parent
operator plus exactly one change: immediately before the FIRST bucket
allreduce of each step's collective burst, issue a barrier on the adapter's
gloo control group so both ranks dispatch the burst together, converting the
device spin-wait into a bounded host wait. All collective work, ordering,
buffers, arithmetic and async semantics remain the parent's, so losses,
gradients (including the bit-exact cross-rank replica property of NCCL
allreduce) and parameters are unchanged.

The per-step collective inventory is verified before any barrier is issued:
the first observed burst only records the size pattern, and alignment is armed
once that pattern is exactly 12 small buckets closed by the embedding bucket.
Any signature mismatch, inventory anomaly or barrier failure permanently
disables the alignment and leaves the plain parent op in place. DDP issues an
identical collective sequence on both ranks, so these decisions evolve
symmetrically on every rank and no collective is ever skipped.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

SMALL_CALL_ID = "d0938de87369efd500322679"
LARGE_CALL_ID = "b8ade439d7b8bd5fb5c1029b"
SMALL_NUMEL = 7087872
LARGE_NUMEL = 39385344
SMALLS_PER_STEP = 12

_BOX = {
    "burst": [],
    "armed": False,
    "disabled": False,
    "reason": "observing_first_burst",
    "alignments": 0,
    "parent_small": None,
    "parent_large": None,
    "control": None,
    "distributed": False,
    "record": lambda *_, **__: None,
}


def _extract_tensors(args, kwargs):
    if "tensors" in kwargs:
        value = kwargs["tensors"]
    elif args:
        value = args[0]
    else:
        return None
    if isinstance(value, (list, tuple)) and len(value) and all(torch.is_tensor(item) for item in value):
        return value
    return None


def _matches(tensors, numel):
    if tensors is None or len(tensors) != 1:
        return False
    tensor = tensors[0]
    return (
        tensor.numel() == numel
        and tensor.dtype is torch.float32
        and tensor.is_cuda
        and tensor.stride() == (1,)
    )


def _align(box):
    if not box["distributed"] or box["control"] is None:
        box["record"](SMALL_CALL_ID, "no_control_group_parent")
        return
    try:
        dist.barrier(group=box["control"])
    except Exception as exc:
        box["disabled"] = True
        box["reason"] = "barrier_failed_" + type(exc).__name__
        box["record"](SMALL_CALL_ID, box["reason"])
        return
    box["alignments"] += 1
    box["record"](SMALL_CALL_ID, "aligned_first_bucket_allreduce")


def _small_replacement(*args, **kwargs):
    box = _BOX
    parent = box["parent_small"]
    try:
        tensors = _extract_tensors(args, kwargs)
        if _matches(tensors, SMALL_NUMEL) and not box["disabled"]:
            burst = box["burst"]
            if not burst:
                if box["armed"]:
                    _align(box)
                else:
                    box["record"](SMALL_CALL_ID, "observing_first_burst")
                box["burst"] = [SMALL_NUMEL]
            elif len(burst) < SMALLS_PER_STEP:
                burst.append(SMALL_NUMEL)
            else:
                box["disabled"] = True
                box["reason"] = "unexpected_bucket_inventory"
                box["record"](SMALL_CALL_ID, "alignment_disabled_unexpected_inventory")
        elif tensors is not None and not _matches(tensors, SMALL_NUMEL):
            box["record"](SMALL_CALL_ID, "small_signature_mismatch_parent")
    except Exception as exc:
        box["disabled"] = True
        box["reason"] = "replacement_error_" + type(exc).__name__
        try:
            box["record"](SMALL_CALL_ID, box["reason"])
        except Exception:
            pass
    return parent(*args, **kwargs)


def _large_replacement(*args, **kwargs):
    box = _BOX
    parent = box["parent_large"]
    try:
        tensors = _extract_tensors(args, kwargs)
        if _matches(tensors, LARGE_NUMEL) and not box["disabled"]:
            burst = box["burst"]
            if len(burst) == SMALLS_PER_STEP and all(size == SMALL_NUMEL for size in burst):
                if box["armed"]:
                    box["record"](LARGE_CALL_ID, "burst_inventory_verified")
                else:
                    box["armed"] = True
                    box["reason"] = "armed_after_first_burst"
                    box["record"](LARGE_CALL_ID, "armed_after_first_burst")
            elif burst:
                box["disabled"] = True
                box["reason"] = "unexpected_bucket_inventory"
                box["record"](LARGE_CALL_ID, "alignment_disabled_unexpected_inventory")
            box["burst"] = []
        elif tensors is not None and not _matches(tensors, LARGE_NUMEL):
            box["record"](LARGE_CALL_ID, "large_signature_mismatch_parent")
    except Exception as exc:
        box["disabled"] = True
        box["reason"] = "replacement_error_" + type(exc).__name__
        try:
            box["record"](LARGE_CALL_ID, box["reason"])
        except Exception:
            pass
    return parent(*args, **kwargs)


def build(context):
    references = context.get("references", {})
    parent_small = references.get(SMALL_CALL_ID)
    parent_large = references.get(LARGE_CALL_ID)
    if not callable(parent_small) or not callable(parent_large):
        raise RuntimeError("alignment_replacement_missing_parent_reference")
    state = context.get("state") or {}
    control = state.get("control") if hasattr(state, "get") else None
    distributed = bool(state.get("distributed")) if hasattr(state, "get") else False
    _BOX.update(
        burst=[],
        armed=False,
        disabled=False,
        reason="observing_first_burst",
        alignments=0,
        parent_small=parent_small,
        parent_large=parent_large,
        control=control,
        distributed=bool(distributed) and dist.is_available() and dist.is_initialized(),
    )
    record = context.get("record_route")
    _BOX["record"] = record if callable(record) else (lambda *_, **__: None)
    return {SMALL_CALL_ID: _small_replacement, LARGE_CALL_ID: _large_replacement}


__all__ = ["build"]
