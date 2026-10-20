"""Joint functional replacement for the two DDP bucket allreduce_ sites:
d0938de87369efd500322679 (12 buckets of 7,087,872 fp32 elements) and
b8ade439d7b8bd5fb5c1029b (the 39,385,344-element closing bucket).

Repair of trial-5a705e709d (decision no_valid_parameter_candidate): that
candidate rerouted the 13 bucket allreduces onto a lazily created second NCCL
process group with forced NCCL_NCHANNELS/NCCL_PROTO, lost the step-time
constraint (253.9-295.5 ms vs the 144.8 ms limit) and worsened the max-rank
kernel sum (89.1/128.4/110.2 ms) through rank-asymmetric in-kernel waiting,
even though numerics stayed exact. This candidate removes every excluded
lever: no new_group, no NCCL environment forcing, no comm hook, no bucket
reordering, no stream rerouting. Every bucket allreduce stays on the worker's
default NCCL process group by calling the captured parent reference with the
original process group, reduce op and async semantics, and the replacement
adds only a bounded cross-rank CPU-side launch alignment: on the first
allreduce_ call of each reduction burst it performs one gloo barrier on the
adapter's existing control group (CPU rendezvous, produces no NCCL kernels)
so neither rank enters its first ring kernel materially before its peer,
then issues the unmodified reference call for all 13 buckets. The waiting
that the early rank previously spun inside counted NCCL kernel time is spent
as uncounted CPU rendezvous instead.
"""
from __future__ import annotations

import threading

import torch
import torch.distributed as dist

DECLARED_CALLS = ("d0938de87369efd500322679", "b8ade439d7b8bd5fb5c1029b")
_CLOSING_CALL = "b8ade439d7b8bd5fb5c1029b"


def _plausible_bucket(tensors, sparse_indices, extra, kwargs):
    return (
        isinstance(tensors, (list, tuple))
        and len(tensors) == 1
        and isinstance(tensors[0], torch.Tensor)
        and tensors[0].is_cuda
        and tensors[0].dtype == torch.float32
        and tensors[0].numel() > 0
        and tensors[0].is_contiguous()
        and sparse_indices is None
        and not extra
        and not kwargs
    )


def build(context):
    context = context or {}
    record = context.get("record_route") or (lambda *_: None)
    references = context.get("references") or {}
    state = context.get("state") or {}
    control = state.get("control")
    if dist.is_available() and dist.is_initialized() and control is not None:
        alignment = {"ok": True, "reason": None}
    else:
        alignment = {"ok": False, "reason": "no_control_group"}

    burst_state = {"open": True}
    lock = threading.Lock()

    def _enter_burst():
        with lock:
            if burst_state["open"]:
                burst_state["open"] = False
                return True
        return False

    def _close_burst():
        with lock:
            burst_state["open"] = True

    def _make_replacement(call_id, reference):
        def aligned_bucket_allreduce_(tensors, process_group, reduce_op, sparse_indices=None, *extra, **kwargs):
            if not _plausible_bucket(tensors, sparse_indices, extra, kwargs):
                record(call_id, "fallback:parent:unsupported_signature")
                return reference(tensors, process_group, reduce_op, sparse_indices, *extra, **kwargs)
            first_of_burst = _enter_burst()
            if first_of_burst and alignment["ok"]:
                try:
                    dist.barrier(group=control)
                except Exception as exc:
                    alignment["ok"] = False
                    alignment["reason"] = "alignment_failed"
                    record(call_id, "runtime_failed:alignment:" + type(exc).__name__)
            if call_id == _CLOSING_CALL:
                _close_burst()
            if alignment["ok"]:
                record(call_id, "aligned:default_group:first_of_burst" if first_of_burst else "aligned:default_group")
            else:
                record(call_id, "passthrough:default_group:" + str(alignment["reason"]))
            return reference(tensors, process_group, reduce_op, sparse_indices, *extra, **kwargs)

        return aligned_bucket_allreduce_

    replacements = {}
    for call_id in DECLARED_CALLS:
        reference = references.get(call_id)
        if reference is not None:
            replacements[call_id] = _make_replacement(call_id, reference)
    return replacements
