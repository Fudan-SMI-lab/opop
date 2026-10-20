"""Joint distributed candidate: fuse the final-step DDP gradient allreduces.

Replaces the reducer's per-bucket c10d::allreduce_ call sites so that every
invocation still performs the real in-place NCCL allreduce on the caller's own
process group and ReduceOp (returning the parent op's own Work object), while
the collectives of one gradient round are issued inside a single NCCL
coalescing group (ProcessGroupNCCL begin/endCoalescing), collapsing the 13
serialized RING_LL launches and their per-kernel arrival windows into one fused
launch over the same 497,759,232-byte logical payload.
"""

from __future__ import annotations

import torch
from torch.distributed.distributed_c10d import ProcessGroup


_LARGE_BUCKET_ELEMENTS = 16_000_000


class _CoalescedRound:
    """One fused round per final-micro-batch backward pass.

    Opens the coalescing group at the round's first bucket, issues every
    bucket's real parent-op allreduce inside it, and closes the group when the
    staged element count reaches the model parameter total (a layout-agnostic
    completion signal). Every returned Work is produced by the parent op itself.
    """

    def __init__(self, total_elements, mode):
        self.total = int(total_elements)
        self.mode = mode
        self.active = True
        self.staged = 0
        self.open = False
        self.pg = None
        self.device = None
        self.group_works = []
        self.busy = False

    def _passthrough(self, reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout):
        return reference(tensors, process_group, reduce_op, sparse_indices, async_op, timeout)

    def _best_effort_close(self):
        try:
            self._close_group()
        except Exception:
            self.open = False

    def _close_group(self):
        try:
            work = self.pg._end_coalescing(self.device)
            if work is not None:
                self.group_works.append(work)
        finally:
            self.open = False

    def __call__(self, reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout):
        if self.busy:
            return self._passthrough(reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout)
        bucket = tensors[0]
        elements = bucket.numel()
        supported = (
            isinstance(bucket, torch.Tensor)
            and bucket.dim() == 1
            and bucket.is_contiguous()
            and bucket.dtype == torch.float32
            and elements > 0
            and bool(async_op)
        )
        if not self.active or not supported:
            return self._passthrough(reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout)
        self.busy = True
        try:
            try:
                pg = ProcessGroup.unbox(process_group)
            except Exception:
                pg = None
            if pg is None:
                self.active = False
                return self._passthrough(reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout)
            self.pg = pg
            self.device = bucket.device
            large = elements >= _LARGE_BUCKET_ELEMENTS

            try:
                if self.open and self.staged + elements > self.total:
                    self._close_group()
                    self.staged = 0
                if large and self.mode == "split_small_large" and self.open and self.staged > 0:
                    self._close_group()
                if not self.open:
                    self.group_works = self.group_works[-2:]
                    self._open_group()
            except Exception:
                self.active = False
                self._best_effort_close()
                return self._passthrough(reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout)

            try:
                result = reference(tensors, process_group, reduce_op, sparse_indices, async_op, timeout)
            except Exception:
                self.active = False
                self._best_effort_close()
                self.staged = 0
                raise

            try:
                self.staged += elements
                if large and self.mode == "split_small_large":
                    self._close_group()
                    if self.staged >= self.total:
                        self.staged = 0
                elif self.staged >= self.total:
                    self._close_group()
                    self.staged = 0
            except Exception:
                self.active = False
                self.staged = 0
            return result
        finally:
            self.busy = False

    def _open_group(self):
        self.pg._start_coalescing(self.device)
        self.open = True


def build(context):
    models = context["models"]
    targets = context["targets"]
    references = context["references"]
    parameters = context.get("parameters") or {}
    mode = parameters.get("trigger_mode", "grouped")
    if mode not in ("grouped", "split_small_large"):
        mode = "grouped"
    total = 0
    for model in models.values():
        total += sum(parameter.numel() for parameter in model.parameters())
    machine = _CoalescedRound(total, mode)

    def make(call_id):
        reference = references[call_id]

        def replacement(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1):
            return machine(reference, tensors, process_group, reduce_op, sparse_indices, async_op, timeout)

        return replacement

    return {call_id: make(call_id) for call_id in targets if call_id in references}
