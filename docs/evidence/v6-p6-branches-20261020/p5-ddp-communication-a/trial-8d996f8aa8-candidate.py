import torch
import torch.distributed as dist

SMALL_BUCKET_CALL = "d0938de87369efd500322679"
LARGE_BUCKET_CALL = "b8ade439d7b8bd5fb5c1029b"


def build(context):
    state = context["state"]
    model = context["models"]["model"]
    references = context.get("references") or {}
    parameters = context.get("parameters") or {}
    trigger_mode = parameters.get("trigger_mode", "single") if isinstance(parameters, dict) else "single"
    if trigger_mode not in ("single", "split_small_large"):
        trigger_mode = "single"

    expected_total = 0
    param_device = None
    try:
        for parameter in model.parameters():
            expected_total += int(parameter.numel())
            param_device = parameter.device
    except Exception:
        expected_total = 0
    if expected_total <= 0 or param_device is None or param_device.type != "cuda":
        return {}

    largest_bucket_numel = 0
    try:
        for spec in (context.get("targets") or {}).values():
            for signature in spec.get("signatures") or []:
                signature_args = signature.get("args") or []
                if not signature_args:
                    continue
                first = signature_args[0]
                if isinstance(first, list) and first and isinstance(first[0], dict):
                    shape = first[0].get("shape") or []
                    if shape:
                        largest_bucket_numel = max(largest_bucket_numel, int(shape[-1]) * len(first))
    except Exception:
        largest_bucket_numel = 0

    thresholds = {expected_total}
    if trigger_mode == "split_small_large" and 0 < largest_bucket_numel < expected_total:
        thresholds.add(expected_total - largest_bucket_numel)

    completed_work = None
    try:
        control = state.get("control")
        if control is not None and dist.is_available() and dist.is_initialized():
            probe = torch.zeros(1, dtype=torch.float32)
            completed_work = dist.all_reduce(probe, group=control, async_op=True)
            completed_work.wait()
    except Exception:
        completed_work = None

    try:
        flat = torch.empty(expected_total, dtype=torch.float32, device=param_device)
    except Exception:
        flat = None

    staging = {"entries": [], "numel": 0, "base": 0}
    active = {"in_flight": False}

    def _extract(args, kwargs):
        def pick(index, name, default):
            if name in kwargs:
                return kwargs[name]
            if len(args) > index:
                return args[index]
            return default

        return (
            pick(0, "tensors", None),
            pick(1, "process_group", None),
            pick(2, "reduce_op", None),
            pick(3, "sparse_indices", None),
            pick(4, "async_op", True),
            pick(5, "timeout", -1),
        )

    def _collective_async(tensors, process_group, reduce_op):
        work = None
        if process_group is not None:
            options_class = getattr(dist, "AllReduceOptions", None)
            if options_class is not None:
                try:
                    options = options_class()
                    try:
                        if reduce_op is not None:
                            options.reduceOp = reduce_op
                    except Exception:
                        pass
                    work = process_group.allreduce(list(tensors), options)
                except Exception:
                    work = None
            if work is None:
                try:
                    work = process_group.allreduce(list(tensors))
                except Exception:
                    work = None
        if work is None:
            if len(tensors) != 1:
                raise RuntimeError("no multi-tensor allreduce fallback available")
            op = reduce_op if reduce_op is not None else dist.ReduceOp.SUM
            work = dist.all_reduce(tensors[0], op=op, group=process_group, async_op=True)
        return work

    def _reduce_direct(tensor, process_group, reduce_op):
        work = _collective_async([tensor], process_group, reduce_op)
        work.wait()
        return work

    def _reset_round():
        staging["entries"] = []
        staging["numel"] = 0
        staging["base"] = 0

    def _drain_staged():
        entries = staging["entries"]
        _reset_round()
        result = None
        for entry in entries:
            work = _reduce_direct(entry["tensor"], entry["process_group"], entry["reduce_op"])
            result = (entry["tensors"], work)
        return result

    def _fire_segment():
        entries = staging["entries"]
        base = staging["base"]
        offset = base
        for entry in entries:
            numel = entry["tensor"].numel()
            flat.narrow(0, offset, numel).copy_(entry["tensor"])
            entry["offset"] = offset
            offset += numel
        last = entries[-1]
        try:
            work = _collective_async(
                [flat.narrow(0, base, offset - base)],
                last["process_group"],
                last["reduce_op"],
            )
        except Exception:
            return _drain_staged()
        work.wait()
        for entry in entries:
            entry["tensor"].copy_(flat.narrow(0, entry["offset"], entry["tensor"].numel()))
        staging["base"] = offset
        staging["entries"] = []
        if staging["numel"] >= expected_total:
            _reset_round()
        return last["tensors"], work

    def _make_replacement(parent):
        def replacement(*args, **kwargs):
            tensors, process_group, reduce_op, sparse_indices, async_op, _timeout = _extract(args, kwargs)
            tensor = None
            if active["in_flight"]:
                if parent is not None:
                    return parent(*args, **kwargs)
                direct = list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]
                work = _collective_async(direct, process_group, reduce_op)
                work.wait()
                return tensors, work
            if (
                flat is not None
                and completed_work is not None
                and sparse_indices is None
                and process_group is not None
                and bool(async_op)
                and isinstance(tensors, (list, tuple))
                and len(tensors) == 1
                and isinstance(tensors[0], torch.Tensor)
            ):
                candidate = tensors[0]
                if (
                    candidate.is_cuda
                    and candidate.device == flat.device
                    and candidate.dtype == torch.float32
                    and candidate.dim() == 1
                    and candidate.is_contiguous()
                    and candidate.numel() > 0
                ):
                    tensor = candidate
            if tensor is None:
                if parent is not None:
                    return parent(*args, **kwargs)
                direct = list(tensors) if isinstance(tensors, (list, tuple)) else [tensors]
                work = _collective_async(direct, process_group, reduce_op)
                work.wait()
                return tensors, work
            numel = tensor.numel()
            if staging["numel"] + numel > expected_total:
                active["in_flight"] = True
                try:
                    _drain_staged()
                    work = _reduce_direct(tensor, process_group, reduce_op)
                finally:
                    active["in_flight"] = False
                return tensors, work
            staging["entries"].append(
                {
                    "tensor": tensor,
                    "tensors": tensors,
                    "process_group": process_group,
                    "reduce_op": reduce_op,
                }
            )
            staging["numel"] += numel
            if staging["numel"] in thresholds:
                active["in_flight"] = True
                try:
                    return _fire_segment()
                finally:
                    active["in_flight"] = False
            return tensors, completed_work

        return replacement

    return {
        SMALL_BUCKET_CALL: _make_replacement(references.get(SMALL_BUCKET_CALL)),
        LARGE_BUCKET_CALL: _make_replacement(references.get(LARGE_BUCKET_CALL)),
    }
