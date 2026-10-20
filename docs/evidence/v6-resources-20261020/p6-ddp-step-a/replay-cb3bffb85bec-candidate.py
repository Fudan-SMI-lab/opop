from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_INVOKE_ID = "5d7d6bb13691b02f533f62b0"
WARMUP_ITERATIONS = 3


def _restore_initial(state, device):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer_state"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)


class _ReplayStep:
    def __init__(self, graph, device, static_loss, static_micro_losses, static_grads):
        self.graph = graph
        self.device = device
        self.static_loss = static_loss
        self.static_micro_losses = static_micro_losses
        self.static_grads = static_grads

    def __call__(self):
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        for parameter, gradient in self.static_grads:
            parameter.grad = gradient
        loss = self.static_loss.clone()
        micro_losses = self.static_micro_losses.clone()
        assert torch.isfinite(loss)
        return {"loss": loss, "micro_losses": micro_losses}


def _build_engine(state, device):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    micro_steps = len(inputs)

    def step_once():
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, tokens in enumerate(inputs):
            is_final = index + 1 == micro_steps
            context = model.no_sync() if isinstance(model, DDP) and not is_final else nullcontext()
            with context:
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / micro_steps
                loss.backward()
                losses.append(loss.detach())
        micro_losses = torch.stack(losses)
        total = micro_losses.sum()
        optimizer.step()
        return total, micro_losses

    torch.cuda.synchronize(device)
    _restore_initial(state, device)

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(WARMUP_ITERATIONS):
            step_once()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    _restore_initial(state, device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss, static_micro_losses = step_once()

    static_grads = [(p, p.grad) for p in state["raw_model"].parameters() if p.grad is not None]
    parameters = list(state["raw_model"].parameters())
    if len(static_grads) != len(parameters):
        raise RuntimeError("capture left parameters without gradient buffers")
    engine = _ReplayStep(graph, device, static_loss, static_micro_losses, static_grads)

    replay_result = engine()
    replay_loss = replay_result["loss"]
    replay_micro = replay_result["micro_losses"]
    replay_params = [p.detach().clone() for p in parameters]
    replay_grads = [g.detach().clone() for _, g in static_grads]

    _restore_initial(state, device)
    eager_loss, eager_micro = step_once()
    torch.cuda.synchronize(device)

    identical = torch.equal(replay_loss, eager_loss) and torch.equal(replay_micro, eager_micro)
    if identical:
        for parameter, saved_param, saved_grad in zip(parameters, replay_params, replay_grads):
            if parameter.grad is None or not torch.equal(parameter.grad, saved_grad) or not torch.equal(parameter.detach(), saved_param):
                identical = False
                break
    del replay_params, replay_grads
    _restore_initial(state, device)
    if not identical:
        raise RuntimeError("cuda graph replay did not reproduce the eager step bitwise")
    return engine


def build(context):
    state = context["state"]
    device = state["device"]
    parent_run = context["references"][PROJECT_INVOKE_ID]

    engine = None
    if device.type == "cuda":
        try:
            engine = _build_engine(state, device)
        except Exception:
            engine = None

    ready = engine is not None
    control = state.get("control")
    if control is not None and dist.is_available() and dist.is_initialized():
        flag = torch.tensor(1 if ready else 0, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=control)
        if not bool(flag.item()):
            engine = None

    def invoke(current_state):
        if engine is None:
            return parent_run(current_state)
        return engine()

    return {PROJECT_INVOKE_ID: invoke}
