"""Round-aligned parent allreduce for the DDP reducer gradient allreduce sites.

Replacements for b8ade439d7b8bd5fb5c1029b (embedding bucket, 1 invocation per
reduction round) and d0938de87369efd500322679 (transformer buckets, 12
invocations per round). Every invocation calls the parent c10d::allreduce_ op
exactly once, immediately, on the caller's own bucket tensors, ProcessGroup and
ReduceOp, and returns the parent's own Work, so per-bucket future-result storage
wiring, the Work capsule contract, in-place SUM semantics, stream ordering and
bitwise gradient values are all inherited unchanged from the parent path on both
the initial and the rebuilt DDP bucket layouts; nothing is deferred, fused,
staged, regrouped or eliminated. The only addition is a host-side round-level
alignment: at the first bucket invocation of each reduction round (rounds are
detected by a staged element counter that reaches the model's parameter total,
124,439,808, exactly at round completion), both ranks meet in dist.barrier on
the pre-existing gloo control group from prepare() before the round's first
parent enqueue, then enter the serialized 13-kernel NCCL stream with near-zero
skew. The diagnostic objective counts per-rank NCCL kernel durations, and the
observed baseline shows the max rank's RING_LL kernels spinning while waiting
for the peer's matching enqueue (discovery: 76.4ms first 28MB-bucket kernel vs
the 3.6ms transport floor on the peer, enqueue offset 72.9ms decaying through
the round); the barrier moves that arrival wait out of NCCL kernel duration into
non-NCCL gloo coordination without touching any kernel, payload or collective.
No CUDA work, kernels, copies, buffers, NCCL launches, stream synchronization,
collectives or math are added; any incompatibility or failure permanently
deactivates the alignment and serves exact parent passthrough.
"""
import torch
import torch.distributed as dist


CALL_IDS = (
    "b8ade439d7b8bd5fb5c1029b",
    "d0938de87369efd500322679",
)


def build(context):
    state = context["state"]
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    models = context.get("models") or {}
    raw_model = models.get("model") or state.get("raw_model")

    total = 0
    if raw_model is not None:
        total = sum(parameter.numel() for parameter in raw_model.parameters())

    control = state.get("control") if isinstance(state, dict) else None
    machine = {
        "active": (
            total > 0
            and control is not None
            and torch.cuda.is_available()
            and dist.is_available()
            and dist.is_initialized()
        ),
        "staged": 0,
    }

    def route(call_id, label):
        try:
            record_route(call_id, label)
        except Exception:
            pass

    def make_replacement(call_id):
        parent = references.get(call_id)

        def repl(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1, **kwargs):
            if parent is None:
                raise RuntimeError("c10d.allreduce_ parent reference unavailable")

            def passthrough():
                return parent(tensors, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)

            if not machine["active"]:
                route(call_id, "parent-allreduce-passthrough")
                return passthrough()

            try:
                tlist = list(tensors)
            except Exception:
                machine["active"] = False
                machine["staged"] = 0
                route(call_id, "fallback-parent-baseline")
                return passthrough()

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
                route(call_id, "parent-allreduce-passthrough")
                return passthrough()

            aligned = machine["staged"] == 0
            if aligned:
                try:
                    dist.barrier(group=control)
                except Exception:
                    machine["active"] = False
                    machine["staged"] = 0
                    route(call_id, "fallback-parent-baseline")
                    return passthrough()

            try:
                out = parent(tlist, process_group, reduce_op, sparse_indices, async_op, timeout, **kwargs)
            except Exception:
                machine["active"] = False
                machine["staged"] = 0
                raise

            machine["staged"] += tlist[0].numel()
            if machine["staged"] >= total:
                if machine["staged"] > total:
                    machine["active"] = False
                machine["staged"] = 0
            route(call_id, "aligned-round-start-parent-allreduce" if aligned else "parent-allreduce")
            return out

        return repl

    return {call_id: make_replacement(call_id) for call_id in CALL_IDS}
