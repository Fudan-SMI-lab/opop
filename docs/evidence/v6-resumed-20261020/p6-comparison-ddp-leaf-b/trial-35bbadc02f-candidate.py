"""Repair of trial-7260b6fac1 for group f833b57a373d8262a384.

The previous attempt crashed because it issued one multi-tensor
c10d::allreduce_([bucket1..bucket12]) list call, which this stack's
ProcessGroupNCCL rejects ("Expecting one tensor only but got multiple").
Every collective issued by this replacement is a SINGLE-TENSOR
c10d::allreduce_ call on one flattened contiguous FP32 staging buffer:
arriving DDP bucket gradients are copied into their segment of a persistent
staging buffer, and at each cohort boundary one flat single-tensor
allreduce_ reduces the gathered segments; the reduced segments are then
scattered back into the stashed bucket tensors (the same storage DDP reduces
in place at baseline) after waiting the genuine Work.
"""

from __future__ import annotations

import torch

CALL_ID = "d0938de87369efd500322679"
DEFAULT_COHORT = 12
FALLBACK_SHAPE = (7087872,)


def build(context):
    spec = (context.get("targets") or {}).get(CALL_ID) or (context.get("all_targets") or {}).get(CALL_ID) or {}
    shape = None
    for signature in spec.get("signatures") or []:
        for arg in signature.get("args", []):
            if (
                isinstance(arg, list)
                and arg
                and isinstance(arg[0], dict)
                and arg[0].get("dtype") == "torch.float32"
                and arg[0].get("device") == "cuda"
            ):
                candidate = tuple(int(v) for v in arg[0]["shape"])
                if len(candidate) == 1:
                    shape = candidate
                    break
        if shape:
            break
    if shape is None:
        shape = FALLBACK_SHAPE
    bucket_elements = shape[0]
    per_step = int(spec.get("count") or DEFAULT_COHORT) or DEFAULT_COHORT

    parameters = dict(context.get("parameters") or {})
    try:
        cohort = int(parameters.get("cohort_size", DEFAULT_COHORT))
    except (TypeError, ValueError):
        cohort = DEFAULT_COHORT
    cohort = max(1, min(cohort, per_step))

    state = context.get("state") or {}
    device = state.get("device")
    if not isinstance(device, torch.device):
        device = torch.device(device if device is not None else "cuda")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    parent = context["references"][CALL_ID]

    staging = torch.zeros(cohort * bucket_elements, dtype=torch.float32, device=device)

    box = {"pending": [], "seed": None, "seed_tensor": None, "step_calls": 0}

    def structure_ok(args, kwargs):
        if len(args) < 4:
            return False
        tensors = args[0]
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            return False
        tensor = tensors[0]
        if not isinstance(tensor, torch.Tensor):
            return False
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            return False
        if tensor.device != device or not tensor.is_contiguous():
            return False
        if kwargs.get("async_op", True) is not True:
            return False
        return True

    def ensure_seed(args, kwargs):
        if box["seed"] is None:
            seed_tensor = torch.zeros(4, dtype=torch.float32, device=device)
            _, work = parent([seed_tensor], *args[1:], **(kwargs or {}))
            work.wait()
            box["seed_tensor"] = seed_tensor
            box["seed"] = work
        return box["seed"]

    def flush(args, kwargs):
        count = len(box["pending"])
        view = staging[: count * bucket_elements]
        _, work = parent([view], *args[1:], **(kwargs or {}))
        work.wait()
        for index, tensor in enumerate(box["pending"]):
            offset = index * bucket_elements
            tensor.copy_(staging[offset : offset + bucket_elements])
        box["pending"].clear()
        return work

    def replacement(*args, **kwargs):
        if not structure_ok(args, kwargs):
            if box["pending"]:
                flush(args, kwargs)
            return parent(*args, **kwargs)
        box["step_calls"] += 1
        seed = ensure_seed(args, kwargs)
        tensors = args[0]
        tensor = tensors[0]
        offset = len(box["pending"]) * bucket_elements
        staging[offset : offset + bucket_elements].copy_(tensor)
        box["pending"].append(tensor)
        end_of_step = box["step_calls"] >= per_step
        if end_of_step:
            box["step_calls"] = 0
        if len(box["pending"]) >= cohort or end_of_step:
            work = flush(args, kwargs)
            return (tensors, work)
        return (tensors, seed)

    return {CALL_ID: replacement}
