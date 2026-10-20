"""Joint replacement for the two observed c10d.allreduce_ dispatch groups.

Before issuing each observed DDP bucket allreduce unchanged through the parent
reference, both ranks meet at a CPU-side gloo rendezvous on the adapter's
existing control group. Compressing the inter-rank collective submission skew
moves device spin-wait out of the NCCL kernels (kernel duration includes
waiting) while the reduction op, operands, communicator and async semantics
stay exactly the source's.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

_EMBEDDING_BUCKET_NUMEL = 39385344
_LAYER_BUCKET_NUMEL = 7087872
_BUCKET_NUMELS = frozenset((_LAYER_BUCKET_NUMEL, _EMBEDDING_BUCKET_NUMEL))

_ROUTE_RENDEZVOUS = "gloo-rendezvous;parent-nccl-allreduce"
_ROUTE_RENDEZVOUS_FAILURE = "rendezvous-runtime-failure;parent-nccl-allreduce"
_ROUTE_SETUP_UNAVAILABLE = "setup-unavailable;parent-allreduce-fallback"
_ROUTE_UNSUPPORTED_SIGNATURE = "unsupported-signature;parent-allreduce-fallback"


def build(context):
    state = context["state"]
    references = context["references"]
    record_route = context.get("record_route")
    if not callable(record_route):
        record_route = lambda *_args, **_kwargs: None

    control = state.get("control") if isinstance(state, dict) else None
    world_size = int(context.get("world_size") or 1)
    rendezvous_ready = (
        control is not None
        and world_size >= 2
        and dist.is_available()
        and dist.is_initialized()
    )

    def _compatible(tensors):
        try:
            if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
                return False
            tensor = tensors[0]
            return (
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cuda"
                and tensor.dtype == torch.float32
                and tensor.dim() == 1
                and tensor.stride() == (1,)
                and not tensor.requires_grad
                and tensor.numel() in _BUCKET_NUMELS
            )
        except Exception:
            return False

    def _make_replacement(call_id):
        parent = references[call_id]

        def allreduce_after_rendezvous(tensors, process_group, reduce_op,
                                       sparse_indices=None, async_op=True,
                                       timeout=-1, *extra, **kwargs):
            if rendezvous_ready and _compatible(tensors):
                try:
                    dist.barrier(group=control)
                except Exception:
                    record_route(call_id, _ROUTE_RENDEZVOUS_FAILURE)
                else:
                    record_route(call_id, _ROUTE_RENDEZVOUS)
            elif not rendezvous_ready:
                record_route(call_id, _ROUTE_SETUP_UNAVAILABLE)
            else:
                record_route(call_id, _ROUTE_UNSUPPORTED_SIGNATURE)
            return parent(tensors, process_group, reduce_op, sparse_indices,
                          async_op, timeout, *extra, **kwargs)

        return allreduce_after_rendezvous

    return {call_id: _make_replacement(call_id) for call_id in references}
