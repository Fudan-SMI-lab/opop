"""Grouped NCCL launch for the twelve DDP layer-bucket gradient allreduces.

Selected site: d0938de87369efd500322679 (c10d.allreduce_.default, the twelve
28.35 MB FP32 layer buckets of DDP(bucket_cap_mb=25, gradient_as_bucket_view=
True) launched by the DDP reducer during the final, gradient-synchronizing
micro-batch). The baseline executes twelve separate latency-oriented
ncclDevKernel_AllReduce_Sum_f32_RING_LL kernels: per-call launch and
rank-meet overhead thirteen times per step, and per-bucket skew stalls.

Policy (identical logic on every rank):

* calls 1..N-1 of a cohort stash their single bucket tensor and return an
  already-completed genuine c10d Work token, leaving the reducer's per-bucket
  bookkeeping intact;
* call N (the last layer bucket of the step) issues ONE c10d.allreduce_ on
  the whole N-tensor list through the parent operator. ProcessGroupNCCL runs
  a multi-tensor allreduce inside a single nccl group, i.e. one large
  contiguous collective that reduces every bucket IN PLACE (the bucket/grad
  aliasing of gradient_as_bucket_view=True is preserved because the results
  land in the original bucket tensors), and returns one real Work;
* that grouped Work is returned for call N, so the reducer's existing
  finalize-time wait keeps exactly the baseline synchronization barrier
  before optimizer.step() reads the gradients. No CPU-side blocking is added.

The token for the very first cohort comes from a one-shot 4-byte allreduce on
the same NCCL process group; afterwards the previous cohort's completed
grouped Work is reused, so steady-state steps launch no extra collectives.
Per-element semantics are unchanged: with the 2-rank elementwise SUM every
element is a single addition of the two rank values regardless of bucket
boundaries, so gradients remain identical across ranks and within reference
tolerances. The sibling embedding-bucket call (b8ade439d7b8bd5fb5c1029b,
group 8525af72c631b94e3dd0) has a different tensor signature and is not part
of this selection: the framework's signature guard routes it to the original
operator, and since both ranks issue the grouped call before the embedding
call, NCCL per-communicator ordering keeps the collective sequences matched.

Any deviation from the observed call shape (extra kwargs, async_op=False,
sparse_indices set, multi/non-contiguous/non-FP32/non-CUDA tensor lists)
forwards unchanged to the parent operator.
"""
from __future__ import annotations

import threading

CALL_ID = "d0938de87369efd500322679"


def build(context):
    import torch

    spec = (context.get("targets") or {}).get(CALL_ID)
    parent = (context.get("references") or {}).get(CALL_ID)
    if spec is None or parent is None:
        return {}
    record_route = context.get("record_route") or (lambda *_: None)
    cohort = int(spec.get("count") or 0)

    def passthrough(tensors, process_group, reduce_op, sparse_indices=None,
                    async_op=True, timeout=-1, **kwargs):
        record_route(CALL_ID, "fallback_parent")
        return parent(tensors, process_group, reduce_op, sparse_indices,
                      async_op=async_op, timeout=timeout, **kwargs)

    if cohort < 2:
        return {CALL_ID: passthrough}

    lock = threading.Lock()
    cell = {"stash": [], "pg": None, "op": None, "seed": None, "done": None}

    def replacement(tensors, process_group, reduce_op, sparse_indices=None,
                    async_op=True, timeout=-1, **kwargs):
        if "async_op" in kwargs:
            async_op = kwargs.pop("async_op")
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")
        try:
            compatible = (
                not kwargs
                and sparse_indices is None
                and bool(async_op)
                and isinstance(tensors, (list, tuple))
                and len(tensors) == 1
                and isinstance(tensors[0], torch.Tensor)
                and tensors[0].is_cuda
                and tensors[0].dim() == 1
                and tensors[0].dtype == torch.float32
                and tensors[0].is_contiguous()
            )
        except Exception:
            compatible = False
        if not compatible:
            return passthrough(tensors, process_group, reduce_op, sparse_indices,
                               async_op, timeout, **kwargs)
        tensor = tensors[0]
        with lock:
            stash = cell["stash"]
            stash.append(tensor)
            if len(stash) == 1:
                cell["pg"] = process_group
                cell["op"] = reduce_op
            if len(stash) < cohort:
                work = cell["done"] or cell["seed"]
                if work is None:
                    probe = torch.zeros(1, dtype=torch.float32, device=tensor.device)
                    cell["seed"] = parent([probe], process_group, reduce_op,
                                          None, True, -1)[1]
                    work = cell["seed"]
                route = "cohort_deferred"
            else:
                grouped, cell["stash"] = stash, []
                result = parent(grouped, cell["pg"], cell["op"], None, True, -1)
                work = result[1] if isinstance(result, (tuple, list)) else result
                cell["done"] = work
                route = "grouped_launch"
        record_route(CALL_ID, route)
        return (tensors, work)

    return {CALL_ID: replacement}
