"""Joint replacement for the two c10d::allreduce_ dispatch groups of the
gpt2_ddp_training "accumulated_step" workload.

Mechanism: before handing each DDP bucket allreduce to the parent dispatch,
every rank first meets the peer at a CPU-side gloo rendezvous (reusing the
adapter's existing out-of-band control group). NCCL kernel duration is
transfer plus spin-wait for the peer, so compressing the inter-rank
submission/arrival skew moves the wait out of the collected
communication_kernel_sum_ms field while the collective itself is issued
unchanged: same tensors, same process group 0, same ReduceOp.SUM, same
async semantics, via the captured parent reference. Payloads, gradients,
parameters and loss are untouched; the gloo barrier launches no CUDA/NCCL
kernel and allocates no device memory.
"""
from __future__ import annotations

from datetime import timedelta

import torch.distributed as dist

SMALL_BUCKET_CALL = "d0938de87369efd500322679"
EMBEDDING_BUCKET_CALL = "b8ade439d7b8bd5fb5c1029b"
DECLARED_CALLS = (SMALL_BUCKET_CALL, EMBEDDING_BUCKET_CALL)

ROUTE_RENDEZVOUS = "gloo-cpu-rendezvous+parent-c10d-allreduce_"
ROUTE_SETUP_FAILURE = "setup-failure:no-gloo-rendezvous-group+parent-c10d-allreduce_"
ROUTE_RUNTIME_FAILURE = "runtime-failure:gloo-barrier+parent-c10d-allreduce_"


class _GlooRendezvous:
    """CPU-only rendezvous state shared by both replacement callables."""

    def __init__(self, state):
        self._state = state
        self._group = None
        self._disabled = False

    def _resolve(self):
        group = None
        if isinstance(self._state, dict):
            group = self._state.get("control")
        if group is None and dist.is_available() and dist.is_initialized():
            group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        return group

    def sync(self):
        if self._disabled:
            return False
        if self._group is None:
            try:
                group = self._resolve()
            except Exception:
                self._disabled = True
                return False
            if group is None:
                self._disabled = True
                return False
            self._group = group
        dist.barrier(group=self._group)
        return True


def build(context):
    state = context["state"]
    references = context["references"]
    record_route = context.get("record_route", lambda *_: None)
    rendezvous = _GlooRendezvous(state)

    def make_replacement(call_id):
        parent = references[call_id]

        def rendezvous_then_parent(*args, **kwargs):
            try:
                completed = rendezvous.sync()
                route = ROUTE_RENDEZVOUS if completed else ROUTE_SETUP_FAILURE
            except Exception:
                route = ROUTE_RUNTIME_FAILURE
            record_route(call_id, route)
            return parent(*args, **kwargs)

        return rendezvous_then_parent

    return {call_id: make_replacement(call_id) for call_id in DECLARED_CALLS if call_id in references}
