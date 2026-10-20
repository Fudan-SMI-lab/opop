"""Joint candidate: reroute the DDP gradient-bucket ``c10d.allreduce_`` calls
through a dedicated, independently initialized NCCL process group.

Measured motivation (inputs/observations, this host): the final backward of each
step issues 13 in-place SUM allreduces on one NCCL stream, all executed as
``ncclDevKernel_AllReduce_Sum_f32_RING_LL`` (12 x 7,087,872 fp32 elements =
28,351,488 B plus 1 x 39,385,344 elements = 157,541,376 B; 497,759,232 B per
rank per step). The isolated calibration measures 3.60-3.68 ms per 28.35 MB
bucket and 19.54-19.83 ms for the 157.5 MB bucket, i.e. only ~7.9-8.0 GB/s
logical payload rate, consistent with the ~50% flag overhead of the LL
protocol on multi-MB payloads over the observed non-P2P SHM path.

Mechanism: a dedicated ``ProcessGroupNCCL`` is constructed over a prefixed view
of the default store so its communicator is created by an independent
``ncclCommInitRank`` (never a split of the parent communicator). The
communicator is initialized while ``NCCL_PROTO`` names a bulk protocol
(default ``SIMPLE``) and with a per-communicator channel cap
(``Options.config.max_ctas``), so the 13 bucket exchanges run the same
in-place SUM with the same async Work contract on a wire-efficient protocol.

Repair of the previous failure (trial-561b4dff10, "missing attribute
capsule"): the dispatcher of ``c10d::allreduce_`` must receive the
``__torch__.torch.classes.c10d.Work`` custom class, not the raw pybind Work
returned by ``ProcessGroupNCCL.allreduce()``. This implementation follows the
pattern torch itself uses in ``torch.distributed._local_tensor._c10d``:
the received ScriptObject holders are consumed with
``reduce_op_so.op()``/``ProcessGroup.unbox`` and the replacement returns
``Work.boxed(work)`` in the (Tensor[], Work) tuple, so the boxed native NCCL
Work satisfies the site ABI and DDP bucket readiness is preserved.

Any setup failure, unsupported signature, or non-bucket shape falls back to
the parent reference; a failure after the collective is enqueued propagates
(never a per-rank divergent fallback).
"""
from __future__ import annotations

import os
from datetime import timedelta

import torch
from torch._C import ScriptObject
from torch._C._distributed_c10d import FakeWork, PrefixStore
import torch.distributed as dist
import torch.distributed.distributed_c10d as _distributed_c10d
from torch.distributed import ProcessGroupNCCL
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp, Work

CALL_IDS = ("b8ade439d7b8bd5fb5c1029b", "d0938de87369efd500322679")
ALT_TIMEOUT = timedelta(seconds=120)
LEGAL_PROTOS = ("SIMPLE", "LL")
FALLBACK_SIZES = frozenset((7087872, 39385344))


def _observed_bucket_sizes(targets):
    sizes = set()
    for call_id in CALL_IDS:
        spec = targets.get(call_id) or {}
        for signature in spec.get("signatures", []):
            args = signature.get("args") or []
            if not args:
                continue
            first = args[0]
            leaves = first if isinstance(first, list) else [first]
            for leaf in leaves:
                if (
                    isinstance(leaf, dict)
                    and leaf.get("dtype") == "torch.float32"
                    and leaf.get("device") == "cuda"
                    and leaf.get("stride") == [1]
                    and leaf.get("shape")
                    and len(leaf["shape"]) == 1
                ):
                    sizes.add(int(leaf["shape"][0]))
    return sizes or set(FALLBACK_SIZES)


def _resolve_reduce_op(value):
    """Extract the real pybind ReduceOp from whatever the dispatcher passes.

    The observed site hands the reduce op over as the scripting-class holder
    ``__torch__.torch.classes.c10d.ReduceOp``; ``holder.op()`` is the accessor
    torch itself uses for this holder (torch.distributed._local_tensor._c10d).
    """
    if isinstance(value, ScriptObject):
        try:
            value = value.op()
        except Exception:
            return None
    if isinstance(value, ReduceOp):
        return value
    return None


def _create_alt_group(proto, max_ctas):
    base_store = _distributed_c10d._get_default_store()
    counter = int(getattr(torch, "_opop_alt_bulk_nccl_builds", 0)) + 1
    torch._opop_alt_bulk_nccl_builds = counter
    store = PrefixStore(f"opop_bulk_nccl:{counter}:{proto}:{max_ctas}", base_store)
    options = ProcessGroupNCCL.Options()
    if max_ctas and max_ctas > 0:
        options.config.max_ctas = int(max_ctas)
    group = ProcessGroupNCCL(store, dist.get_rank(), dist.get_world_size(), options)
    # Force lazy communicator initialization while NCCL_PROTO is applied, and
    # verify the alternative executor end to end with a one-element probe.
    saved = os.environ.get("NCCL_PROTO")
    os.environ["NCCL_PROTO"] = proto
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        probe = torch.zeros(1, dtype=torch.float32, device=device)
        work = group.allreduce([probe])
        work.wait()
        torch.cuda.synchronize()
        if float(probe.item()) != 0.0:
            raise RuntimeError("alternative group probe returned a nonzero sum")
    finally:
        if saved is None:
            os.environ.pop("NCCL_PROTO", None)
        else:
            os.environ["NCCL_PROTO"] = saved
    return group


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    references = context.get("references") or {}
    targets = context.get("targets") or {}
    parameters = dict(context.get("parameters") or {})

    proto = str(parameters.get("nccl_proto", "SIMPLE")).strip().upper()
    if proto not in LEGAL_PROTOS:
        proto = "SIMPLE"
    try:
        max_ctas = int(parameters.get("nccl_max_ctas", 0))
    except (TypeError, ValueError):
        max_ctas = 0

    allowed_sizes = _observed_bucket_sizes(targets)

    group = None
    try:
        world_size = int(context.get("world_size") or dist.get_world_size())
        if not (dist.is_available() and dist.is_initialized()):
            group = None
        elif world_size != 2:
            group = None
        else:
            group = _create_alt_group(proto, max_ctas)
    except Exception:
        group = None

    route_alt = f"alt_nccl_{proto.lower()}_ctas{max_ctas}"

    def _make_replacement(call_id):
        parent = references[call_id]

        def replace_allreduce_(*args, **kwargs):
            tensors = args[0] if args else kwargs.get("tensors")
            reduce_op_arg = args[2] if len(args) > 2 else kwargs.get("reduce_op")
            sparse_indices = args[3] if len(args) > 3 else kwargs.get("sparse_indices")
            async_op = args[4] if len(args) > 4 else kwargs.get("async_op", True)

            if group is None:
                record_route(call_id, "parent_fallback_setup")
                return parent(*args, **kwargs)

            eligible = (
                isinstance(tensors, (list, tuple))
                and len(tensors) == 1
                and isinstance(tensors[0], torch.Tensor)
                and tensors[0].device.type == "cuda"
                and tensors[0].dtype is torch.float32
                and tensors[0].dim() == 1
                and tensors[0].stride() == (1,)
                and not tensors[0].requires_grad
                and tensors[0].numel() in allowed_sizes
                and sparse_indices is None
            )
            if not eligible:
                record_route(call_id, "parent_fallback_signature")
                return parent(*args, **kwargs)

            reduce_op = _resolve_reduce_op(reduce_op_arg)
            if reduce_op is None:
                record_route(call_id, "parent_fallback_signature")
                return parent(*args, **kwargs)

            work = group.allreduce(list(tensors), reduce_op, ALT_TIMEOUT)
            if not async_op:
                work.wait()
            record_route(call_id, route_alt)
            # The dispatcher must receive the __torch__.torch.classes.c10d.Work
            # custom class; Work.boxed() converts the pybind Work (the pattern
            # torch.distributed._local_tensor._c10d uses for these c10d ops).
            work_holder = Work.boxed(work) if work is not None else Work.boxed(FakeWork())
            return (tensors, work_holder)

        return replace_allreduce_

    replacements = {}
    for call_id in CALL_IDS:
        if call_id in targets and call_id in references:
            replacements[call_id] = _make_replacement(call_id)
    return replacements
