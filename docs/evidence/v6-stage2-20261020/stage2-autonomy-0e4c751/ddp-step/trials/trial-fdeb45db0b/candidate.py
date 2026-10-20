"""Coordinated batching policy for the two DDP gradient-allreduce dispatch sites.

The frozen workload allreduces 13 DDP gradient buckets on the final micro-step
(12 contiguous fp32 buckets of 7,087,872 elements and one of 39,385,344
elements), each dispatched through ``c10d::allreduce_`` with async_op=True.
Both dispatch sites are replaced with one shared policy that keeps every
dispatch site, every bucket tensor and the SUM semantics unchanged while
consolidating the collectives:

* small-bucket dispatches are deferred; the call still returns a real c10d
  Work handle (the most recently issued collective's Work, or a one-element
  scratch allreduce issued through the parent operator on the very first
  deferral, when no earlier Work exists yet);
* when the large-bucket dispatch arrives (observed last on both ranks), the
  accumulated buffers are appended to it and issued as one multi-tensor
  collective through the parent-class coalescing operator
  ``c10d.allreduce_coalesced_`` (ProcessGroupNCCL fuses the per-tensor
  ncclAllReduce enqueues of one nccl group into a single grouped kernel;
  ``c10d.allreduce_`` itself rejects tensor lists longer than one on the
  NCCL backend);
* a tunable ``small_group_size`` flushes queued small buckets earlier, trading
  consolidation for the baseline's partial overlap of early allreduces with
  the final backward.

Ordering safety: every trigger is call-count based and identical on all ranks
(the bucket readiness order of the frozen model is deterministic), so the
process-group collective sequence stays matched. Numerics: with two ranks the
per-element reduction is a single commutative fp32 addition, so a grouped call
is bitwise equal to the per-bucket calls.

Why deferral is safe with torch 2.13 DDP: the C++ caller of this op is
``ProcessGroup::allreduce`` invoked from ``_AllReduceBySumCommHook::runHook``,
which only takes the returned Work's future; the reducer stores those futures
and waits for all of them (``Reducer::finalize_backward``: collectAll ->
wait -> finalize_bucket_dense) before backward returns. The folded collective
issued at the large-bucket dispatch is the last collective of the step on the
process group, so its Work is the synchronization point covering every
deferred bucket's data, exactly like the baseline's final bucket Work.
"""
from __future__ import annotations

import threading

import torch

SMALL_SITE = "d0938de87369efd500322679"
LARGE_SITE = "b8ade439d7b8bd5fb5c1029b"

try:
    _COALESCED_OP = torch.ops.c10d.allreduce_coalesced_
except AttributeError:  # pragma: no cover - very old builds without the op
    _COALESCED_OP = None


class _AllreduceBatcher:
    """State machine shared by both replacement callables (per-rank, per-install)."""

    def __init__(self, group_size: int, record):
        self.group_size = max(2, int(group_size))
        self._record = record
        self._pending: list[torch.Tensor] = []
        self._last_work = None
        self._scratch = None
        self._lock = threading.Lock()
        self._in_flight = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _eligible(tensors) -> bool:
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return False
        tensor = tensors[0]
        return (isinstance(tensor, torch.Tensor) and tensor.is_cuda
                and tensor.dtype == torch.float32 and tensor.dim() == 1
                and tensor.is_contiguous())

    @staticmethod
    def _unpack(args, kwargs):
        tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
        pg = args[1] if len(args) > 1 else kwargs.get("process_group")
        reduce_op = args[2] if len(args) > 2 else kwargs.get("reduce_op")
        sparse = args[3] if len(args) > 3 else kwargs.get("sparse_indices")
        async_op = args[4] if len(args) > 4 else kwargs.get("async_op", True)
        timeout = args[5] if len(args) > 5 else kwargs.get("timeout", -1)
        return tensors, pg, reduce_op, sparse, async_op, timeout

    def _issue(self, reference, tensors, pg, reduce_op, sparse, timeout):
        """Issue one collective through the parent operators.

        Single tensors go through the intercepted call's own parent operator
        (c10d.allreduce_, matching the baseline dispatch exactly). Multiple
        tensors use c10d.allreduce_coalesced_, the parent-class coalescing
        operator for same-device lists (ProcessGroupNCCL::allreduce_coalesced
        wraps the per-tensor ncclAllReduce enqueue in one nccl group, fusing
        the list into a single grouped kernel); c10d.allreduce_ itself rejects
        tensor lists longer than one on the NCCL backend.

        _in_flight defeats same-thread re-entry: if an outer dispatch mode
        (e.g. a discovery capture or a stacked binding) routes this very call
        back into a replacement, that re-entrant invocation must pass straight
        through to the parent so the collective is issued once.
        """
        self._in_flight = True
        try:
            if len(tensors) > 1 and _COALESCED_OP is not None:
                work = _COALESCED_OP(list(tensors), pg, reduce_op, True, timeout)
            elif len(tensors) > 1:
                raise RuntimeError("coalesced allreduce operator unavailable")
            else:
                result = reference(list(tensors), pg, reduce_op, sparse)
                if not (isinstance(result, (list, tuple)) and len(result) == 2):
                    raise RuntimeError("parent c10d.allreduce_ returned no (tensors, Work) pair")
                work = result[1]
        finally:
            self._in_flight = False
        self._last_work = work
        return work

    def _issue_grouped(self, reference, tensors, pg, reduce_op, sparse, timeout):
        """Issue one grouped call; on failure degrade to per-tensor parent calls."""
        try:
            return self._issue(reference, tensors, pg, reduce_op, sparse, timeout), False
        except Exception:
            work = None
            for tensor in tensors:
                work = self._issue(reference, [tensor], pg, reduce_op, sparse, timeout)
            return work, True

    def _shared_work(self, reference, tensors, pg, reduce_op, sparse, timeout):
        """A real Work handle for a deferred call: the previous collective's Work.

        On the very first deferral of an install no earlier Work exists, so a
        one-element scratch allreduce is issued through the parent operator
        (once per install; enqueued before any grouped call, on every rank).
        """
        if self._last_work is None:
            if self._scratch is None:
                self._scratch = torch.zeros(1, dtype=torch.float32, device=tensors[0].device)
            self._issue(reference, [self._scratch], pg, reduce_op, sparse, timeout)
        return self._last_work

    def _flush(self, reference, pg, reduce_op, sparse, timeout, extra=()):
        tensors = self._pending + list(extra)
        self._pending = []
        return self._issue_grouped(reference, tensors, pg, reduce_op, sparse, timeout)

    def _passthrough(self, reference, args, kwargs):
        self._in_flight = True
        try:
            result = reference(*args, **kwargs)
        finally:
            self._in_flight = False
        if isinstance(result, (list, tuple)) and len(result) == 2:
            self._last_work = result[1]
        return result

    def _defer_eligible(self, tensors, async_op):
        return self._eligible(tensors) and bool(async_op)

    # ------------------------------------------------------------------ sites
    def small(self, call_id, reference, args, kwargs):
        if self._in_flight:
            return self._passthrough(reference, args, kwargs)
        tensors, pg, reduce_op, sparse, async_op, timeout = self._unpack(args, kwargs)
        if not self._defer_eligible(tensors, async_op):
            with self._lock:
                flushed = bool(self._pending)
                if flushed:
                    self._flush(reference, pg, reduce_op, sparse, timeout)
                result = self._passthrough(reference, args, kwargs)
            self._record(call_id, "fallback_parent_after_flush" if flushed else "fallback_parent")
            return result
        with self._lock:
            pointer = tensors[0].data_ptr()
            if any(existing.data_ptr() == pointer for existing in self._pending):
                # Anomalous repeated bucket storage (no flush fired last step):
                # reduce what is queued now, order-preserving on every rank.
                self._flush(reference, pg, reduce_op, sparse, timeout)
            self._pending.extend(tensors)
            if len(self._pending) >= self.group_size:
                work, degraded = self._flush(reference, pg, reduce_op, sparse, timeout)
                route = "flush_grouped" if not degraded else "flush_error_per_tensor"
            else:
                try:
                    work = self._shared_work(reference, tensors, pg, reduce_op, sparse, timeout)
                    route = "deferred_batched"
                except Exception:
                    work, degraded = self._flush(reference, pg, reduce_op, sparse, timeout)
                    route = "setup_defer_unavailable" if not degraded else "flush_error_per_tensor"
        self._record(call_id, route)
        return (tensors, work)

    def large(self, call_id, reference, args, kwargs):
        if self._in_flight:
            return self._passthrough(reference, args, kwargs)
        tensors, pg, reduce_op, sparse, async_op, timeout = self._unpack(args, kwargs)
        if not self._defer_eligible(tensors, async_op):
            with self._lock:
                flushed = bool(self._pending)
                if flushed:
                    self._flush(reference, pg, reduce_op, sparse, timeout)
                result = self._passthrough(reference, args, kwargs)
            self._record(call_id, "fallback_parent_after_flush" if flushed else "fallback_parent")
            return result
        with self._lock:
            if self._pending:
                work, degraded = self._flush(reference, pg, reduce_op, sparse, timeout, extra=tensors)
                route = "flush_folded_large" if not degraded else "flush_error_per_tensor"
            else:
                work, degraded = self._issue_grouped(reference, tensors, pg, reduce_op, sparse, timeout)
                route = "solo_parent" if not degraded else "flush_error_per_tensor"
        self._record(call_id, route)
        return (tensors, work)


def build(context):
    parameters = context.get("parameters") or {}
    group_size = int(parameters.get("small_group_size", 16))
    record = context.get("record_route")
    if not callable(record):
        def record(call_id, route):
            return None
    batcher = _AllreduceBatcher(group_size, record)
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    replacements = {}
    if SMALL_SITE in targets and SMALL_SITE in references and LARGE_SITE in targets and LARGE_SITE in references:
        small_reference = references[SMALL_SITE]
        large_reference = references[LARGE_SITE]

        def small_replacement(*args, **kwargs):
            return batcher.small(SMALL_SITE, small_reference, args, kwargs)

        def large_replacement(*args, **kwargs):
            return batcher.large(LARGE_SITE, large_reference, args, kwargs)

        replacements[SMALL_SITE] = small_replacement
        replacements[LARGE_SITE] = large_replacement
    return replacements
