"""Repair of trial-b0a02885e2: joint two-rank bucket-reduction replacement.

Replaces the DDP bucket allreduce dispatch at the two selected c10d.allreduce_
call sites with a bandwidth-oriented two-rank exchange on the same
worker-initialized process group. The previous attempt never executed its
exchange because its reduce-op guard rejected the ScriptObject argument this
build of torch hands to Python replacements; this repair identifies SUM via
the accessor torch itself uses (torch/distributed/_local_tensor/_c10d.py:
``reduce_op_so.op()``), resolves the boxed ProcessGroup via
``ProcessGroup.unbox``, and supports two exchange strategies selected by the
tunable ``exchange`` parameter:

* ``p2p``: chunk-free one-shot halves exchange -- each rank isends one
  contiguous half, irecvs the peer's half into persistent scratch, reduces its
  kept half with one local add, and swaps the reduced halves back with a
  second grouped exchange (SIMPLE-protocol bulk point-to-point transfers).
* ``rs_ag``: reduce-scatter + all-gather decomposition via the process
  group's pybind collectives (``_reduce_scatter_base``/``_allgather_base``),
  the canonical equivalent at the same wire volume.

Every output element is one two-addend commutative fp32 sum computed on
exactly one rank and then copied to the peer, so results are bitwise equal to
the parent SUM and bitwise identical across ranks; DDP's world-size division
happens outside this boundary and is untouched. Guards are rank-symmetric and
run strictly before anything is enqueued; exceptions after the first enqueue
propagate instead of triggering a parent fallback.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import P2POp
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    ProcessGroup,
    ReduceOp,
    ReduceScatterOptions,
    Work,
)

CALL_IDS = (
    "d0938de87369efd500322679",
    "b8ade439d7b8bd5fb5c1029b",
)

_MIN_ELEMENTS = 2 * 1024 * 1024
_EXCHANGE_DEVICES = ("cuda",)
_EXCHANGE_BACKENDS = ("nccl",)
_EXCHANGE_MODES = ("p2p", "rs_ag")
_DEFAULT_EXCHANGE = "p2p"
_TAG_STRIDE = 2
_TAG_SPACE = 1 << 30


def _resolve_process_group(process_group):
    if process_group is None:
        return dist.group.WORLD
    if isinstance(process_group, ProcessGroup):
        return process_group
    try:
        return ProcessGroup.unbox(process_group)
    except Exception:
        return None


def _reduce_op_is_sum(reduce_op):
    if reduce_op is None:
        return False
    redoptype = getattr(ReduceOp, "RedOpType", None)
    if isinstance(reduce_op, ReduceOp):
        try:
            return bool(reduce_op == ReduceOp.SUM)
        except Exception:
            return False
    if redoptype is not None and isinstance(reduce_op, redoptype):
        try:
            return bool(reduce_op == ReduceOp.SUM)
        except Exception:
            return False
    getter = getattr(reduce_op, "op", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return False
        try:
            return bool(value == ReduceOp.SUM)
        except Exception:
            return False
    return False


def _plan_exchange(tensors, process_group, reduce_op, sparse_indices, async_op,
                   scratch_for):
    try:
        if sparse_indices is not None:
            return "guard_sparse_indices"
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return "guard_tensor_list"
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type not in _EXCHANGE_DEVICES:
            return "guard_device"
        if tensor.dtype != torch.float32:
            return "guard_dtype"
        if tensor.dim() != 1 or not tensor.is_contiguous():
            return "guard_layout"
        numel = tensor.numel()
        if numel < _MIN_ELEMENTS or numel % 2 != 0:
            return "guard_size"
        if not async_op:
            return "guard_async_op"
        pg = _resolve_process_group(process_group)
        if pg is None:
            return "setup_process_group"
        try:
            backend = str(dist.get_backend(pg))
        except Exception:
            return "setup_backend"
        if backend not in _EXCHANGE_BACKENDS:
            return "guard_backend"
        try:
            ranks = dist.get_process_group_ranks(pg)
        except Exception:
            return "setup_ranks"
        if len(ranks) != 2:
            return "guard_world_size"
        if not _reduce_op_is_sum(reduce_op):
            return "guard_reduce_op"
        group_rank = dist.get_rank(pg)
        if group_rank not in (0, 1):
            return "guard_rank"
        half = numel // 2
        return {
            "tensors": tensors,
            "tensor": tensor,
            "pg": pg,
            "peer": ranks[1 - group_rank],
            "first_half_owned": group_rank == 0,
            "half": half,
            "scratch": scratch_for(tensor.device, half),
        }
    except Exception as exc:
        return "error_" + type(exc).__name__


def _execute_p2p(plan, tag_base):
    tensor = plan["tensor"]
    pg = plan["pg"]
    peer = plan["peer"]
    half = plan["half"]
    scratch = plan["scratch"]
    send_region = tensor[:half] if plan["first_half_owned"] else tensor[half:]
    keep_region = tensor[half:] if plan["first_half_owned"] else tensor[:half]
    tag1 = tag_base % _TAG_SPACE
    tag2 = (tag_base + 1) % _TAG_SPACE
    works1 = dist.batch_isend_irecv([
        P2POp(dist.isend, send_region, peer, group=pg, tag=tag1),
        P2POp(dist.irecv, scratch, peer, group=pg, tag=tag1),
    ])
    for work in works1:
        work.wait()
    keep_region.add_(scratch)
    works2 = dist.batch_isend_irecv([
        P2POp(dist.isend, keep_region, peer, group=pg, tag=tag2),
        P2POp(dist.irecv, send_region, peer, group=pg, tag=tag2),
    ])
    final = None
    for work in works2:
        work.wait()
        final = work
    return (plan["tensors"], Work.boxed(final))


def _execute_rs_ag(plan, opts):
    tensor = plan["tensor"]
    out_half = plan["scratch"]
    work_rs = plan["pg"]._reduce_scatter_base(out_half, tensor, opts["rs"])
    work_rs.wait()
    work_ag = plan["pg"]._allgather_base(tensor, out_half, opts["ag"])
    work_ag.wait()
    return (plan["tensors"], Work.boxed(work_ag))


def build(context):
    parameters = dict(context.get("parameters") or {})
    exchange = str(parameters.get("exchange", _DEFAULT_EXCHANGE)).lower()
    if exchange not in _EXCHANGE_MODES:
        exchange = _DEFAULT_EXCHANGE
    record_route = context.get("record_route") or (lambda *_: None)
    references = context["references"]
    scratch_cache = {}
    tag_state = [0]
    exchange_opts = {}

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
            plan = _plan_exchange(tensors, process_group, reduce_op,
                                  sparse_indices, async_op, scratch_for)
            if isinstance(plan, str):
                record_route(call_id, "parent:" + plan)
                return parent(tensors, process_group, reduce_op,
                              sparse_indices, async_op, timeout, **kwargs)
            tag_base = tag_state[0] * _TAG_STRIDE
            tag_state[0] += 1
            record_route(call_id, exchange + "_oneshot")
            if exchange == "rs_ag":
                if "rs" not in exchange_opts:
                    rs_opts = ReduceScatterOptions()
                    rs_opts.reduceOp = ReduceOp.SUM
                    exchange_opts["rs"] = rs_opts
                    exchange_opts["ag"] = AllgatherOptions()
                return _execute_rs_ag(plan, exchange_opts)
            return _execute_p2p(plan, tag_base)

        return replacement

    return {call_id: make_replacement(call_id) for call_id in CALL_IDS if call_id in references}
