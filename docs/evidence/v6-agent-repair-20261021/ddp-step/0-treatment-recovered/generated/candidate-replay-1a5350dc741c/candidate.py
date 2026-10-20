"""Joint replacement for the two selected DDP bucket allreduce dispatch sites.

Selection: groups f833b57a373d8262a384 (call d0938de87369efd500322679, 12
invocations of the 7,087,872-element bucket) and 8525af72c631b94e3dd0 (call
b8ade439d7b8bd5fb5c1029b, the single 39,385,344-element embedding bucket), both
at the shared ``c10d.allreduce_.default`` functional boundary.

Repair of trial-b0a02885e2, whose replacement never executed because the
reduce-op guard rejected the dispatcher's argument: at this op the ReduceOp
arrives as a ``torch.ScriptObject`` wrapping ``__torch__.torch.classes.c10d.ReduceOp``
whose ``op()`` returns the RedOpType value as a plain int (observed 0 for SUM),
and ``torch._C._distributed_c10d.RedOpType`` is not importable at top level in
this torch layout (it is nested as ``type(ReduceOp.SUM)``). The guard now
accepts the boxed form only when the script class identity matches exactly and
the value converts to the RedOpType enum and equals SUM; arbitrary integers
are never treated as SUM. Verified against the installed worker torch
2.13.0+cu129 on CPU (``torch.classes.c10d.ReduceOp().op() -> 0``,
``int(ReduceOp.SUM) == 0``, ``type(ReduceOp.SUM)(0) == ReduceOp.SUM``).

Implementation: the selection hypothesis's explicitly permitted "equivalent
path". Instead of host-orchestrated isend/irecv rounds (two host-ordered
rounds plus a local add per bucket), the SUM allreduce of each flat fp32
bucket on the *same* worker-initialized NCCL process group is decomposed into

  * ``reduce_scatter_single(scratch_half, bucket, SUM, async_op=True)`` and
  * ``all_gather_single(bucket, scratch_half, async_op=True)``,

both enqueued back-to-back on the process group's NCCL stream, which orders
the allgather after the reduce-scatter without any host wait. The callable
returns ``(tensors, work)`` with the allgather work handle boxed via
``work.boxed()`` - the same return convention torch's own c10d meta
registrations use - preserving async semantics at this op's exact ABI.

Wire volume per rank is S bytes (each rank sends its half in each direction),
the information-theoretic minimum for an exact allreduce and identical to the
parent ring allreduce; the measured parent path is NCCL RING_LL at ~8.05 GB/s
logical, and the decomposition lets NCCL schedule the two half-size stages
independently. Every output element is a single two-addend commutative fp32
add computed on exactly one rank (the owner of that half in the ring
reduce-scatter) and then copied, so results are bitwise identical across
ranks and to the parent SUM (DDP's world-size division happens before this
boundary and is untouched).

Guards are rank-symmetric (identical bucket shape/dtype/layout, backend,
world membership and reduce op on both ranks); any unsupported case falls
back to the parent op *before* enqueueing anything, so the two ranks can
never take divergent collective paths. Exceptions raised after the first
enqueue are never converted into a parent fallback (that would double-reduce
the bucket); they propagate and the run fails loudly instead.

Known measured context (previous-feedback.txt): the prior attempt executed 13
parent fallbacks with byte-identical outputs (max_abs_error 0.0) and still
measured 230.5/234.1 ms against parent 131.0/139.1 ms, i.e. the legacy
functional binding itself - a global dispatch mode active for the whole
measured invocation - costs ~+95 ms on this workload, more than the ~62 ms
total communication these sites can influence. This repair makes the selected
hypothesis actually executable so the paired measurement can decide it; the
binding cost itself is outside a replacement's control.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp

CALL_IDS = (
    "d0938de87369efd500322679",
    "b8ade439d7b8bd5fb5c1029b",
)

# Backends/devices for which the split reduction is eligible.
_SPLIT_DEVICES = ("cuda",)
_SPLIT_BACKENDS = ("nccl",)

_MIN_ELEMENTS = 2 * 1024 * 1024

_REDUCE_OP_SCRIPT_CLASS = "c10d.ReduceOp"

# In this torch layout the RedOpType enum is nested inside the pybind ReduceOp
# class (torch._C._distributed_c10d.ReduceOp.RedOpType); there is no top-level
# import, which is what broke the previous candidate's guard.
_RedOpEnum = type(ReduceOp.SUM)


def _resolve_process_group(process_group):
    """Return the pybind ProcessGroup for the op's argument, or None."""
    if process_group is None:
        return dist.group.WORLD
    if isinstance(process_group, ProcessGroup):
        return process_group
    try:
        return ProcessGroup.unbox(process_group)
    except Exception:
        return None


def _reduce_op_is_sum(reduce_op):
    """True only when the argument is positively identified as ReduceOp.SUM.

    The dispatcher marshals the op's ReduceOp argument as a torch.ScriptObject
    wrapping __torch__.torch.classes.c10d.ReduceOp; op() returns the RedOpType
    value as a plain int. Accept that form only when the script class identity
    matches the c10d ReduceOp custom class and the value converts through the
    RedOpType enum and equals SUM. Plain integers of unknown provenance are
    never accepted.
    """
    if reduce_op is None:
        return True
    if isinstance(reduce_op, _RedOpEnum):
        return reduce_op == ReduceOp.SUM
    class_of = getattr(reduce_op, "_type", None)
    value_of = getattr(reduce_op, "op", None)
    if callable(class_of) and callable(value_of):
        try:
            name = str(class_of())
            if not name.endswith(_REDUCE_OP_SCRIPT_CLASS):
                return False
            return _RedOpEnum(value_of()) == ReduceOp.SUM
        except Exception:
            return False
    return False


def _plan_split(tensors, process_group, reduce_op, sparse_indices, scratch_for):
    """Validate the invocation and resolve the exchange context. Returns a
    reason string for parent fallback, or a plan dict. Performs no
    communication and allocates scratch lazily; every failure here happens
    before anything is enqueued."""
    try:
        if sparse_indices is not None:
            return "guard_sparse_indices"
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return "guard_tensor_list"
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type not in _SPLIT_DEVICES:
            return "guard_device"
        if tensor.dtype != torch.float32:
            return "guard_dtype"
        if tensor.dim() != 1 or not tensor.is_contiguous():
            return "guard_layout"
        numel = tensor.numel()
        if numel < _MIN_ELEMENTS or numel % 2 != 0:
            return "guard_size"
        pg = _resolve_process_group(process_group)
        if pg is None:
            return "setup_process_group"
        try:
            backend = str(dist.get_backend(pg))
        except Exception:
            return "setup_backend"
        if backend not in _SPLIT_BACKENDS:
            return "guard_backend"
        try:
            ranks = dist.get_process_group_ranks(pg)
        except Exception:
            return "setup_ranks"
        if len(ranks) != 2:
            return "guard_world_size"
        if not _reduce_op_is_sum(reduce_op):
            return "guard_reduce_op"
        return {
            "pg": pg,
            "tensor": tensor,
            "half": scratch_for(tensor.device, numel // 2),
            # NCCL executes collectives enqueued on one process group in issue
            # order on the group's single device stream, so the allgather is
            # device-ordered after the reduce-scatter. Backends without that
            # guarantee (e.g. a CPU test harness) take an explicit wait.
            "stream_ordered": backend == "nccl",
        }
    except Exception as exc:  # pre-enqueue failure; safe to fall back
        return "error_" + type(exc).__name__


def _execute_split(plan, tensors):
    """Enqueue reduce-scatter + allgather. No parent fallback past this point
    by design. On the guarded nccl path both collectives run on the process
    group's single NCCL stream, which executes ops in issue order, so the
    allgather is ordered after the reduce-scatter without any host wait; the
    returned allgather work therefore covers the entire exchange. Backends
    without issue-order execution take an explicit wait between the two."""
    reduced = dist.reduce_scatter_single(plan["half"], plan["tensor"],
                                         op=ReduceOp.SUM, group=plan["pg"], async_op=True)
    if not plan["stream_ordered"]:
        reduced.wait()
    work = dist.all_gather_single(plan["tensor"], plan["half"],
                                   group=plan["pg"], async_op=True)
    return (tensors, work.boxed())


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context["references"]
    scratch_cache = {}

    def scratch_for(device, half_elements):
        key = (device.type, device.index)
        entry = scratch_cache.get(key)
        if entry is None or entry.numel() < half_elements:
            entry = torch.empty(half_elements, dtype=torch.float32, device=device)
            scratch_cache[key] = entry
        return entry[:half_elements]

    def make_replacement(call_id):
        parent = references[call_id]

        def replacement(tensors, process_group=None, reduce_op=None,
                        sparse_indices=None, async_op=True, timeout=-1, **kwargs):
            if kwargs or async_op is not True:
                record_route(call_id, "parent:guard_signature")
                return parent(tensors, process_group, reduce_op,
                              sparse_indices, async_op, timeout, **kwargs)
            plan = _plan_split(tensors, process_group, reduce_op,
                               sparse_indices, scratch_for)
            if isinstance(plan, str):
                record_route(call_id, "parent:" + plan)
                return parent(tensors, process_group, reduce_op,
                              sparse_indices, async_op, timeout)
            record_route(call_id, "split_allreduce")
            return _execute_split(plan, tensors)

        return replacement

    return {call_id: make_replacement(call_id) for call_id in CALL_IDS if call_id in references}
