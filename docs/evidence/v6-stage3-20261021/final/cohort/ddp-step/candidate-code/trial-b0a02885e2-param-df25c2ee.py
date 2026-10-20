"""Joint replacement for the two selected DDP bucket allreduce dispatch sites.

Selection: groups f833b57a373d8262a384 (call d0938de87369efd500322679, 12
invocations of the 7,087,872-element bucket) and 8525af72c631b94e3dd0 (call
b8ade439d7b8bd5fb5c1029b, the single 39,385,344-element embedding bucket), both
at the shared ``c10d.allreduce_.default`` functional boundary.

The replacement implements the selected hypothesis as a chunked one-shot
two-rank exchange on the *same* worker-initialized process group that the
parent op receives:

  * the flat fp32 bucket view is split into two contiguous halves;
  * round 1 exchanges the halves with grouped NCCL isend/irecv (both
    directions move bulk contiguous transfers) into a persistent scratch
    buffer and the local add produces the reduced half;
  * round 2 swaps the reduced halves back into the bucket view;
  * the callable returns ``(tensors, Work)`` with the same async semantics as
    the parent op, boxing the final irecv work handle.

Every output element is produced by exactly one two-addend commutative fp32
add on exactly one rank and then copied to the peer, so cross-rank values are
bitwise identical by construction and equal to the parent SUM (DDP's own
world-size division happens before this boundary and is untouched).

Guards are rank-symmetric (identical bucket shapes/dtype/backend/reduce-op on
both ranks); any unsupported case falls back to the parent op *before*
enqueueing anything, so the two ranks can never take divergent collective
paths. Exceptions raised after the first enqueue are never converted into a
parent fallback (that would double-reduce the bucket); they propagate and the
run fails loudly instead.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import P2POp
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp, Work

CALL_IDS = (
    "d0938de87369efd500322679",
    "b8ade439d7b8bd5fb5c1029b",
)

# Devices/backends for which the one-shot exchange is eligible. Kept as module
# constants so CPU-side API tests can exercise the same code path.
_P2P_DEVICES = ("cuda",)
_P2P_BACKENDS = ("nccl",)

_MIN_ELEMENTS = 2 * 1024 * 1024
_MIN_CHUNK_ELEMENTS = 262144
_DEFAULT_CHUNKS = 8
_MAX_CHUNKS = 64
_TAG_STRIDE = 2
_TAG_SPACE = 1 << 30

try:
    from torch._C._distributed_c10d import RedOpType
except Exception:  # pragma: no cover - very old/new torch layouts
    RedOpType = None


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
    """True only when the argument is positively identified as ReduceOp.SUM."""
    if reduce_op is None:
        return True
    values = []
    if isinstance(reduce_op, ReduceOp):
        values.append(reduce_op)
    elif hasattr(reduce_op, "op"):
        try:
            values.append(reduce_op.op())
        except Exception:
            pass
    else:
        values.append(reduce_op)
    for value in values:
        if isinstance(value, ReduceOp) and value == ReduceOp.SUM:
            return True
        if RedOpType is not None and isinstance(value, RedOpType) and value == RedOpType.SUM:
            return True
    return False


def _plan_exchange(tensors, process_group, reduce_op, sparse_indices, chunks,
                   scratch_for, tag_base):
    """Validate and pre-build the exchange. Returns a reason string for parent
    fallback, or a plan dict. Performs no communication and allocates scratch
    lazily; every failure here happens before anything is enqueued."""
    try:
        if sparse_indices is not None:
            return "guard_sparse_indices"
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return "guard_tensor_list"
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type not in _P2P_DEVICES:
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
        if backend not in _P2P_BACKENDS:
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
        peer = ranks[1 - group_rank]
        half = numel // 2
        n_chunks = max(1, min(chunks, half // _MIN_CHUNK_ELEMENTS or 1))
        scratch = scratch_for(tensor.device, half)
        send_region = tensor[:half] if group_rank == 0 else tensor[half:]
        keep_region = tensor[half:] if group_rank == 0 else tensor[:half]
        send_chunks = list(torch.chunk(send_region, n_chunks))
        keep_chunks = list(torch.chunk(keep_region, n_chunks))
        scratch_chunks = list(torch.chunk(scratch, n_chunks))
        tag1 = tag_base % _TAG_SPACE
        tag2 = (tag_base + 1) % _TAG_SPACE
        round1 = [P2POp(dist.isend, chunk, peer, group=pg, tag=tag1) for chunk in send_chunks]
        round1 += [P2POp(dist.irecv, chunk, peer, group=pg, tag=tag1) for chunk in scratch_chunks]
        round2 = [P2POp(dist.isend, chunk, peer, group=pg, tag=tag2) for chunk in keep_chunks]
        round2 += [P2POp(dist.irecv, chunk, peer, group=pg, tag=tag2) for chunk in send_chunks]
        return {
            "n_chunks": n_chunks,
            "round1": round1,
            "round2": round2,
            "keep_chunks": keep_chunks,
            "scratch_chunks": scratch_chunks,
        }
    except Exception as exc:  # pre-enqueue failure; safe to fall back
        return "error_" + type(exc).__name__


def _execute_exchange(plan, tensors):
    """Enqueue the exchange. No parent fallback past this point by design."""
    works1 = dist.batch_isend_irecv(plan["round1"])
    for work in works1:
        work.wait()
    for keep_chunk, scratch_chunk in zip(plan["keep_chunks"], plan["scratch_chunks"]):
        keep_chunk.add_(scratch_chunk)
    works2 = dist.batch_isend_irecv(plan["round2"])
    final = None
    for work in works2:
        work.wait()
        final = work
    return (tensors, Work.boxed(final))


def build(context):
    parameters = dict(context.get("parameters") or {})
    try:
        chunks = int(parameters.get("chunks", _DEFAULT_CHUNKS))
    except (TypeError, ValueError):
        chunks = _DEFAULT_CHUNKS
    chunks = max(1, min(_MAX_CHUNKS, chunks))

    record_route = context.get("record_route") or (lambda *_: None)
    references = context["references"]
    scratch_cache = {}
    tag_state = [0]

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
            tag_base = tag_state[0] * _TAG_STRIDE
            tag_state[0] += 1
            plan = _plan_exchange(tensors, process_group, reduce_op,
                                  sparse_indices, chunks, scratch_for, tag_base)
            if isinstance(plan, str):
                record_route(call_id, "parent:" + plan)
                return parent(tensors, process_group, reduce_op,
                              sparse_indices, async_op, timeout)
            record_route(call_id, "p2p_oneshot")
            return _execute_exchange(plan, tensors)

        return replacement

    return {call_id: make_replacement(call_id) for call_id in CALL_IDS if call_id in references}
