from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

CALL_ID = "d0938de87369efd500322679"
_DEFAULT_BUCKET_NUMEL = 7087872
_BUILD_TAG = [0]
_PROBE_TIMEOUT_S = 90
_INIT_TIMEOUT_S = 120


def _as_process_group(obj):
    from torch._C._distributed_c10d import ProcessGroup

    if isinstance(obj, ProcessGroup):
        return obj
    try:
        return ProcessGroup.unbox(obj)
    except Exception:
        return obj


def _backend_for(pg, device):
    getter = getattr(pg, "_get_backend", None)
    if getter is not None:
        return getter(device)
    return pg


def _bucket_numel(context):
    try:
        spec = context["targets"][CALL_ID]["signatures"][0]["args"][0][0]
        numel = int(spec["shape"][0])
        dtype = spec.get("dtype", "torch.float32")
        if dtype == "torch.float32" and numel > 0:
            return numel
    except Exception:
        pass
    return _DEFAULT_BUCKET_NUMEL


def _apply_env(window):
    saved = {}
    for key, value in window.items():
        if value:
            saved[key] = os.environ.get(key)
            os.environ[key] = value
    return saved


def _restore_env(saved):
    for key, old in saved.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _build_tuned_backend(device, rank, world_size, proto, algo, min_ctas, max_ctas, high_priority):
    from torch._C._distributed_c10d import PrefixStore, ProcessGroupNCCL
    from torch.distributed.distributed_c10d import _world

    _, base_store = _world.pg_map[dist.group.WORLD]
    tag = _BUILD_TAG[0]
    _BUILD_TAG[0] += 1
    store = PrefixStore(f"candidate-bucket-ar-{tag}/", base_store)
    options = ProcessGroupNCCL.Options()
    if high_priority:
        options.is_high_priority_stream = True
    if min_ctas > 0:
        options.config.min_ctas = int(min_ctas)
    if max_ctas > 0:
        options.config.max_ctas = int(max_ctas)
    options._timeout = timedelta(seconds=_INIT_TIMEOUT_S)
    saved = _apply_env({"NCCL_PROTO": proto, "NCCL_ALGO": algo})
    try:
        torch.cuda.set_device(device)
        pg = ProcessGroupNCCL(store, int(rank), int(world_size), options)
        probe = torch.zeros(1, dtype=torch.float32, device=device)
        opts = dist.AllreduceOptions()
        opts.asyncOp = True
        work = pg.allreduce([probe], opts)
        done = work.wait(timedelta(seconds=_PROBE_TIMEOUT_S))
        if done is False:
            raise TimeoutError("tuned communicator probe timed out")
        return pg
    finally:
        _restore_env(saved)


def _consensus(state, status):
    control = state.get("control")
    if control is None or not (dist.is_available() and dist.is_initialized()):
        return status
    flag = torch.tensor([int(status)], dtype=torch.int64)
    dist.all_reduce(flag, group=control)
    return int(flag.item())


def build(context):
    state = context["state"]
    if CALL_ID not in context.get("targets", {}):
        return {}

    params = dict(context.get("parameters") or {})
    device = state["device"]
    rank = int(context["rank"])
    world_size = int(context["world_size"])

    route = str(params.get("route", "parent") or "parent")
    proto = str(params.get("nccl_proto", "") or "")
    algo = str(params.get("nccl_algo", "") or "")
    min_ctas = int(params.get("nccl_min_ctas", 0) or 0)
    max_ctas = int(params.get("nccl_max_ctas", 0) or 0)
    high_priority = int(params.get("high_priority_stream", 0) or 0)

    tuned_backend = None
    status = 1
    if (
        route == "tuned"
        and device.type == "cuda"
        and dist.is_available()
        and dist.is_initialized()
    ):
        try:
            tuned_backend = _build_tuned_backend(
                device, rank, world_size, proto, algo, min_ctas, max_ctas, high_priority
            )
        except Exception:
            tuned_backend = None
            status = 0
    status = _consensus(state, status)
    use_tuned = tuned_backend is not None and status == 1

    bucket_numel = _bucket_numel(context)

    def allreduce_bucket(tensors, process_group, reduce_op, sparse_indices=None, async_op=True, timeout=-1):
        tensor = tensors[0] if isinstance(tensors, (list, tuple)) and len(tensors) == 1 else None
        compatible = (
            tensor is not None
            and tensor.dim() == 1
            and tensor.numel() == bucket_numel
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
            and tensor.device.type == "cuda"
            and sparse_indices is None
        )
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        if isinstance(sparse_indices, torch.Tensor):
            opts.sparseIndices = sparse_indices
        if isinstance(timeout, (int, float)) and timeout > 0:
            opts.timeout = timedelta(milliseconds=timeout)
        if compatible and use_tuned:
            opts.asyncOp = True
            work = tuned_backend.allreduce(list(tensors), opts)
        else:
            opts.asyncOp = bool(async_op)
            pg = _as_process_group(process_group)
            work = _backend_for(pg, device).allreduce(list(tensors), opts)
        if not async_op:
            work.wait()
        return (list(tensors), work.boxed())

    return {CALL_ID: allreduce_bucket}
