"""Joint distributed candidate for groups f833b57a373d8262a384 / 8525af72c631b94e3dd0.

Before issuing each observed DDP gradient-bucket NCCL allreduce unchanged through
its parent reference, both ranks meet in a CPU-side gloo rendezvous on the
adapter's existing out-of-band control group, so the two ranks submit each of
the 13 collectives nearly simultaneously instead of with the observed ~15-16 ms
stagger. The rendezvous launches no CUDA/NCCL kernel, carries no payload, and
the parent c10d::allreduce_ call (same tensors, process group 0, ReduceOp.SUM,
async semantics) is forwarded verbatim, so reduction results are unchanged.
"""
import torch
import torch.distributed as dist

_EXPECTED_NUMEL = {
    "d0938de87369efd500322679": 7087872,
    "b8ade439d7b8bd5fb5c1029b": 39385344,
}

_ROUTE_ALIGNED = "rendezvous+parent-nccl-allreduce"
_ROUTE_SIGNATURE = "fallback:signature-mismatch"
_ROUTE_NO_GROUP = "passthrough:no-rendezvous-group"
_ROUTE_RENDEZVOUS_ERROR = "fallback:rendezvous-error"


def _bucket_tensor_matches(value, numel):
    if not isinstance(value, torch.Tensor):
        return False
    return (
        value.is_cuda
        and value.dtype == torch.float32
        and tuple(value.shape) == (numel,)
        and tuple(value.stride()) == (1,)
    )


def _signature_matches(tensors, numel):
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
        return False
    return _bucket_tensor_matches(tensors[0], numel)


def _scalar_signature_matches(args, kwargs):
    async_op = kwargs.get("async_op", args[4] if len(args) > 4 else True)
    timeout = kwargs.get("timeout", args[5] if len(args) > 5 else -1)
    return async_op is True and timeout == -1


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route") or (lambda *_: None)
    world_size = int(context.get("world_size") or 1)

    control = state.get("control")
    rendezvous_state = {
        "enabled": (
            control is not None
            and world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        ),
        "control": control,
        "failures": 0,
    }

    def rendezvous():
        if not rendezvous_state["enabled"]:
            return False
        try:
            dist.barrier(group=rendezvous_state["control"])
            return True
        except Exception:
            rendezvous_state["enabled"] = False
            rendezvous_state["failures"] += 1
            return False

    def make_replacement(call_id):
        numel = _EXPECTED_NUMEL[call_id]
        parent = references[call_id]

        def replacement(*args, **kwargs):
            tensors = args[0] if args else kwargs.get("tensors")
            if not (
                _signature_matches(tensors, numel)
                and _scalar_signature_matches(args, kwargs)
            ):
                record_route(call_id, _ROUTE_SIGNATURE)
                return parent(*args, **kwargs)
            aligned = rendezvous()
            if aligned:
                record_route(call_id, _ROUTE_ALIGNED)
            elif rendezvous_state["failures"]:
                record_route(call_id, _ROUTE_RENDEZVOUS_ERROR)
            else:
                record_route(call_id, _ROUTE_NO_GROUP)
            return parent(*args, **kwargs)

        return replacement

    return {
        call_id: make_replacement(call_id)
        for call_id in _EXPECTED_NUMEL
        if call_id in references
    }
