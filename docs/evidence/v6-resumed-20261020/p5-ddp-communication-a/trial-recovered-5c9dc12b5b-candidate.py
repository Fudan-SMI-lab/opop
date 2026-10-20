"""Readiness-pinned coalesced launch for the DDP reducer gradient allreduce sites.

Replacements for b8ade439d7b8bd5fb5c1029b (embedding bucket, 1 invocation) and
d0938de87369efd500322679 (transformer buckets, 12 invocations): every
invocation still calls the parent c10d::allreduce_ op on the caller's own
ProcessGroup and ReduceOp with the caller's own bucket tensors and returns the
parent op's own Work, so per-bucket future-result storage wiring, the Work
capsule contract, in-place SUM semantics and bitwise gradient values are all
inherited from the parent path. The whole reduction round is additionally
issued inside one ProcessGroupNCCL coalescing scope (_start_coalescing at the
round's first bucket, _end_coalescing at round completion), consolidating the
13 per-bucket launches into one fused group launch. The previous failure of
this mechanism - deterministic corruption of the wte region of the embedding
bucket - is repaired by pinning readiness with host-side current-stream
synchronization before every op enqueue and again before _end_coalescing
launches the fused group, so the grouped kernels can never read bucket
contents before the final micro-batch's gradient writes (including
embedding_dense_backward) are GPU-complete, regardless of which internal
ordering was deficient. No extra kernels, copies, buffers, collectives or
math are introduced; any incompatibility deactivates the mechanism and serves
exact parent passthrough.
"""
from contextlib import contextmanager

import torch

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:

    @contextmanager
    def _phase(name):
        yield


CALL_IDS = (
    "b8ade439d7b8bd5fb5c1029b",
    "d0938de87369efd500322679",
)


def _unbox(process_group):
    try:
        from torch._C._distributed_c10d import ProcessGroup

        wrapper = ProcessGroup.unbox(process_group)
        if hasattr(wrapper, "_start_coalescing"):
            return wrapper
    except Exception:
        pass
    return None


def build(context):
    state = context["state"]
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    models = context.get("models") or {}
    raw_model = models.get("model") or state.get("raw_model")

    total = 0
    if raw_model is not None:
        total = sum(parameter.numel() for parameter in raw_model.parameters())

    machine = {
        "active": total > 0 and torch.cuda.is_available(),
        "open": False,
        "staged": 0,
        "wrapper": None,
        "device": None,
        "flushed_work": None,
    }

    def make_replacement(call_id):
        parent = references.get(call_id)

        def repl(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1, **kwargs):
            if parent is None:
                raise RuntimeError("c10d.allreduce_ parent reference unavailable")

            def passthrough():
                return parent(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)

            if not machine["active"]:
                return passthrough()

            try:
                tlist = list(tensors)
                compatible = (
                    sparse_indices is None
                    and bool(async_op)
                    and len(tlist) == 1
                    and tlist[0].is_cuda
                    and tlist[0].dtype is torch.float32
                    and tlist[0].is_contiguous()
                    and tlist[0].dim() == 1
                )
                if not compatible:
                    return passthrough()

                device = tlist[0].device
                if not machine["open"]:
                    wrapper = (
                        process_group
                        if hasattr(process_group, "_start_coalescing")
                        else _unbox(process_group)
                    )
                    if wrapper is None:
                        machine["active"] = False
                        return passthrough()
                    try:
                        wrapper._start_coalescing(device)
                    except Exception:
                        machine["active"] = False
                        return passthrough()
                    machine["open"] = True
                    machine["staged"] = 0
                    machine["wrapper"] = wrapper
                    machine["device"] = device

                torch.cuda.current_stream(device).synchronize()
                out = parent(tlist, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)
                machine["staged"] += tlist[0].numel()
                staged_now = machine["staged"]
                record_route(call_id, "coalesced-group-allreduce")
                if staged_now >= total:
                    torch.cuda.current_stream(device).synchronize()
                    try:
                        machine["flushed_work"] = machine["wrapper"]._end_coalescing(machine["device"])
                    except Exception:
                        machine["active"] = False
                    machine["open"] = False
                    machine["staged"] = 0
                    if staged_now > total:
                        machine["active"] = False
                return out
            except Exception:
                if machine["open"]:
                    try:
                        machine["wrapper"]._end_coalescing(machine["device"])
                    except Exception:
                        pass
                machine["open"] = False
                machine["staged"] = 0
                machine["active"] = False
                record_route(call_id, "fallback-parent-baseline")
                return passthrough()

        return repl

    return {call_id: make_replacement(call_id) for call_id in CALL_IDS}
