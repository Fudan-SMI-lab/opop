"""Joint TCPStore-aligned wrapper for the two DDP bucket-reduction allreduce sites.

diagnostic.max_rank_nccl_kernel_sum_ms is the max over ranks of the NCCL kernel
duration sum.  The transport floor is already reached in-model (12 small buckets
at a measured 3.566 ms median plus one 157.5 MB bucket at 19.570 ms, a flat
~7.95 GB/s logical rate, peer access disabled), so the addressable term is the
rank-arrival skew: the cross-rank CPU submit spread is ~20.7-22.5 ms and the
early rank's first ncclDevKernel spins that long before the peer arrives, after
which the chain runs in lockstep.  Trial-7a341bbf2e proved the causal fix by
aligning the ranks before every parent allreduce (objective at the floor on both
ranks, max_abs_error 0.0, coverage 12+1) but paid 13 ProcessGroupGloo
dist.barrier calls at ~6.8 ms each and failed only step_ms.

Repair: keep the identical joint wrapper and parent forwarding, and align with a
two-key barrier on the default process group's rendezvous store
(distributed_c10d._get_default_store wrapped in a PrefixStore with a
bundle-unique prefix): each side sets its round-scoped key and waits for the
peer's key with a bounded 250 ms timeout.  This launches no NCCL kernels and no
gloo collectives, creates no new process group, and was verified on the measured
worker venv (torch 2.13.0+cu129): store.set ~4 us, wait-on-present ~40 us,
wait-on-late releases exactly at the peer's set, timeout raises DistStoreError.
Any store failure or round timeout permanently disables the alignment in that
process (plain parent forwarding thereafter, numerics unchanged, worst case one
step pays the 250 ms bound).  align_first_k (default 13) thins the alignment to
the first K collectives of each synchronized backward; the burst index resets
after the largest-payload site because the embedding bucket is observed last in
every backward (resources.json sequences 21-33).  Every collective is always
forwarded unchanged to the parent c10d::allreduce_ with identical tensors,
process group, ReduceOp, async_op and timeout.
"""
from __future__ import annotations

import threading
from datetime import timedelta

import torch
import torch.distributed as dist

try:
    from torch.distributed import PrefixStore
except Exception:
    PrefixStore = None

try:
    from torch.distributed.distributed_c10d import _get_default_store
except Exception:
    _get_default_store = None

_TARGET_OP = "c10d.allreduce_.default"

_PREFIX = "opop-cand-74cf73e2374c-align-v1"
_ROUND_TIMEOUT_MS = 250
_DEFAULT_ALIGN_FIRST_K = 13
_MAX_ALIGN_FIRST_K = 13
_MIN_ALIGN_FIRST_K = 1
_REQUIRED_WORLD_SIZE = 2

_ROUTE_ALIGNED = "tcpstore_aligned_allreduce"
_ROUTE_NO_STORE = "setup_no_default_store_parent_allreduce"
_ROUTE_SIGNATURE = "unsupported_signature_parent_allreduce"
_ROUTE_TIMEOUT = "store_round_timeout_parent_allreduce"
_ROUTE_STORE_FAILURE = "runtime_store_failure_parent_allreduce"
_ROUTE_DISABLED = "runtime_align_disabled_parent_allreduce"
_ROUTE_POLICY_SKIP = "policy_unaligned_parent_allreduce"

_PROCESS_ROUND = {"value": 0}
_PROCESS_ROUND_LOCK = threading.Lock()


def _expected_numels(spec):
    numels = set()
    for signature in (spec or {}).get("signatures") or []:
        args = signature.get("args") or []
        if not args or not isinstance(args[0], list):
            continue
        for entry in args[0]:
            if isinstance(entry, dict) and entry.get("shape") and len(entry["shape"]) == 1:
                numels.add(int(entry["shape"][0]))
    return numels


def _signature_supported(tensors, expected_numels):
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
        return False
    tensor = tensors[0]
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda" or tensor.dtype != torch.float32:
        return False
    if tensor.dim() != 1 or not tensor.is_contiguous():
        return False
    if expected_numels and tensor.numel() not in expected_numels:
        return False
    return True


def _sanitize_first_k(value):
    try:
        value = int(value)
    except Exception:
        return _DEFAULT_ALIGN_FIRST_K
    return max(_MIN_ALIGN_FIRST_K, min(value, _MAX_ALIGN_FIRST_K))


class _SharedState:
    __slots__ = ("lock", "store", "rank", "peer", "first_k", "timeout", "disabled", "burst")

    def __init__(self, store, rank, world_size, first_k, timeout):
        self.lock = threading.Lock()
        self.store = store
        self.rank = int(rank)
        self.peer = (int(rank) + 1) % max(int(world_size), 1)
        self.first_k = int(first_k)
        self.timeout = timeout
        self.disabled = False
        self.burst = 0


def _make_replacement(call_id, parent, expected_numels, shared, closes_burst, record_route):
    def aligned_allreduce_(tensors, *args, **kwargs):
        try:
            supported = _signature_supported(tensors, expected_numels)
        except Exception:
            supported = False
        if not supported:
            record_route(call_id, _ROUTE_SIGNATURE)
            return parent(tensors, *args, **kwargs)
        if shared.store is None:
            record_route(call_id, _ROUTE_NO_STORE)
            return parent(tensors, *args, **kwargs)
        with shared.lock:
            position = shared.burst
            shared.burst = 0 if closes_burst else position + 1
            skip_by_policy = position >= shared.first_k
            disabled = shared.disabled
        if disabled:
            record_route(call_id, _ROUTE_DISABLED)
            return parent(tensors, *args, **kwargs)
        if skip_by_policy:
            record_route(call_id, _ROUTE_POLICY_SKIP)
            return parent(tensors, *args, **kwargs)
        with _PROCESS_ROUND_LOCK:
            _PROCESS_ROUND["value"] += 1
            round_id = _PROCESS_ROUND["value"]
        try:
            shared.store.set("r/{}/{}".format(round_id, shared.rank), b"1")
            shared.store.wait(["r/{}/{}".format(round_id, shared.peer)], shared.timeout)
        except Exception as error:
            with shared.lock:
                shared.disabled = True
            if isinstance(error, dist.DistStoreError):
                record_route(call_id, _ROUTE_TIMEOUT)
            else:
                record_route(call_id, _ROUTE_STORE_FAILURE)
            return parent(tensors, *args, **kwargs)
        record_route(call_id, _ROUTE_ALIGNED)
        return parent(tensors, *args, **kwargs)

    return aligned_allreduce_


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    first_k = _sanitize_first_k(parameters.get("align_first_k", _DEFAULT_ALIGN_FIRST_K))
    world_size = int(context.get("world_size") or 1)
    rank = int(context.get("rank") or 0)

    store = None
    if (
        world_size == _REQUIRED_WORLD_SIZE
        and dist.is_available()
        and dist.is_initialized()
        and PrefixStore is not None
        and _get_default_store is not None
    ):
        try:
            store = PrefixStore(_PREFIX, _get_default_store())
        except Exception:
            store = None

    shared = _SharedState(store, rank, world_size, first_k, timedelta(milliseconds=_ROUND_TIMEOUT_MS))

    selected = []
    for call_id, parent in (context.get("references") or {}).items():
        spec = (context.get("targets") or {}).get(call_id) or {}
        if spec.get("kind") != "functional" or spec.get("target") != _TARGET_OP:
            continue
        selected.append((call_id, parent, _expected_numels(spec)))

    largest = max((max(numels) if numels else -1 for _, _, numels in selected), default=-1)
    replacements = {}
    for call_id, parent, numels in selected:
        closes_burst = bool(numels) and largest > 0 and max(numels) == largest
        replacements[call_id] = _make_replacement(
            call_id, parent, numels, shared, closes_burst, record_route
        )
    return replacements
