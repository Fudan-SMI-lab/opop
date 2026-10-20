from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_INVOKE_CALL = "5d7d6bb13691b02f533f62b0"
_ALIGNMENT = 16


def _optimizer_parameters(state):
    unique = {}
    for group in state["optimizer"].param_groups:
        for parameter in group["params"]:
            unique.setdefault(id(parameter), parameter)
    return list(unique.values())


def _flat_views(parameters, device):
    offsets = []
    offset = 0
    for parameter in parameters:
        offsets.append(offset)
        offset += (parameter.numel() + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    flat = torch.zeros(offset, dtype=parameters[0].dtype, device=device)
    views = [flat[off : off + parameter.numel()].view_as(parameter) for parameter, off in zip(parameters, offsets)]
    return flat, views


def _parameter_layout_supported(parameters, device):
    if not parameters:
        return False
    for parameter in parameters:
        if parameter.device != device or parameter.dtype != torch.float32:
            return False
        if not parameter.is_contiguous():
            return False
    return True


def _restore_state(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer_state"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], state["device"])


def _ranks_agree(state, local_ok):
    control = state.get("control")
    world = int(state.get("world_size", 1))
    if world <= 1 or control is None:
        return bool(local_ok)
    try:
        flags = [None] * world
        dist.all_gather_object(flags, bool(local_ok), group=control)
        return all(flags)
    except Exception:
        return False


class _GraphPlan:
    def __init__(self, parameters, views, flat, static_inputs, graph_compute, graph_update, micro, total, world, device):
        self.parameters = parameters
        self.views = views
        self.flat = flat
        self.static_inputs = static_inputs
        self.graph_compute = graph_compute
        self.graph_update = graph_update
        self.micro = micro
        self.total = total
        self.world = world
        self.device = device

    def run(self, state):
        inputs = list(state["inputs"])
        if len(inputs) != len(self.static_inputs):
            raise RuntimeError("input count differs from captured workload")
        for parameter, view in zip(self.parameters, self.views):
            parameter.grad = view
        for static, tokens in zip(self.static_inputs, inputs):
            static.copy_(tokens)
        self.graph_compute.replay()
        if self.world > 1 and dist.is_available() and dist.is_initialized():
            work = dist.all_reduce(self.flat, op=dist.ReduceOp.SUM, async_op=True)
            work.wait()
        assert torch.isfinite(self.total)
        self.graph_update.replay()
        micro = self.micro.clone()
        total = self.total.clone()
        torch.cuda.synchronize(self.device)
        return {"loss": total, "micro_losses": micro}


class _EagerPlan:
    def __init__(self, raw_model, optimizer, parameters, views, flat, world, device):
        self.raw_model = raw_model
        self.optimizer = optimizer
        self.parameters = parameters
        self.views = views
        self.flat = flat
        self.world = world
        self.device = device

    def run(self, state):
        inputs = list(state["inputs"])
        count = len(inputs)
        for parameter, view in zip(self.parameters, self.views):
            parameter.grad = view
        self.flat.zero_()
        losses = []
        for tokens in inputs:
            loss = self.raw_model(input_ids=tokens, labels=tokens, use_cache=False).loss / count
            loss.backward()
            losses.append(loss.detach())
        if any(parameter.grad is not view for parameter, view in zip(self.parameters, self.views)):
            offset = 0
            with torch.no_grad():
                for parameter in self.parameters:
                    self.flat[offset : offset + parameter.numel()].copy_(parameter.grad.reshape(-1))
                    offset += (parameter.numel() + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            for parameter, view in zip(self.parameters, self.views):
                parameter.grad = view
        if self.world > 1 and dist.is_available() and dist.is_initialized():
            work = dist.all_reduce(self.flat, op=dist.ReduceOp.SUM, async_op=True)
            work.wait()
        self.flat.div_(self.world)
        micro = torch.stack(losses)
        total = micro.sum()
        assert torch.isfinite(total)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"loss": total, "micro_losses": micro}


def _mirror_run(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        is_final = i + 1 == len(inputs)
        context = model.no_sync() if isinstance(model, DDP) and not is_final else nullcontext()
        with context:
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
            loss.backward()
            losses.append(loss.detach())
    total = torch.stack(losses).sum()
    micro = torch.stack(losses)
    assert torch.isfinite(total)
    optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize(state["device"])
    return {"loss": total, "micro_losses": micro}


def _build_graph_plan(state):
    device = state["device"]
    if device.type != "cuda":
        raise RuntimeError("graph plan requires cuda")
    raw_model = state["raw_model"]
    optimizer = state["optimizer"]
    inputs = list(state["inputs"])
    count = len(inputs)
    parameters = _optimizer_parameters(state)
    if not _parameter_layout_supported(parameters, device):
        raise RuntimeError("unsupported parameter layout")
    world = int(state["world_size"])
    flat, views = _flat_views(parameters, device)
    for parameter, view in zip(parameters, views):
        parameter.grad = view
    static_inputs = [tokens.detach().clone() for tokens in inputs]

    def compute():
        flat.zero_()
        losses = []
        for static in static_inputs:
            output = raw_model(input_ids=static, labels=static, use_cache=False)
            loss = output.loss / count
            loss.backward()
            losses.append(loss.detach())
        micro = torch.stack(losses)
        total = micro.sum()
        return micro, total

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            compute()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    pool = torch.cuda.graph_pool_handle()
    graph_compute = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_compute, pool=pool):
        micro, total = compute()
    graph_update = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_update, pool=pool):
        for parameter, view in zip(parameters, views):
            parameter.grad = view
        flat.div_(world)
        optimizer.step()
    if not all(parameter.grad is view for parameter, view in zip(parameters, views)):
        raise RuntimeError("gradient views were reassigned during capture")
    return _GraphPlan(parameters, views, flat, static_inputs, graph_compute, graph_update, micro, total, world, device)


def _build_eager_plan(state):
    device = state["device"]
    parameters = _optimizer_parameters(state)
    if not _parameter_layout_supported(parameters, device):
        raise RuntimeError("unsupported parameter layout")
    flat, views = _flat_views(parameters, device)
    return _EagerPlan(state["raw_model"], state["optimizer"], parameters, views, flat, int(state["world_size"]), device)


def build(context):
    state = context["state"]
    graph_plan = None
    try:
        graph_plan = _build_graph_plan(state)
    except Exception:
        graph_plan = None
    _restore_state(state)
    if not _ranks_agree(state, graph_plan is not None):
        graph_plan = None
    plan = graph_plan
    if plan is None:
        try:
            plan = _build_eager_plan(state)
        except Exception:
            plan = None

    def invoke(current_state):
        if plan is not None:
            return plan.run(current_state)
        return _mirror_run(current_state)

    return {PROJECT_INVOKE_CALL: invoke}
