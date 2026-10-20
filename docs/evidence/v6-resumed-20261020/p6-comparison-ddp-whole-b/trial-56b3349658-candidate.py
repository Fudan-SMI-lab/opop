from __future__ import annotations

import gc
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CALL_ID = "5d7d6bb13691b02f533f62b0"
GRAD_ATOL = 1e-06
GRAD_RTOL = 1e-04
PARAM_ATOL = 1e-06
PARAM_RTOL = 1e-04


def _reset_like(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer_state"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng_state"])
    if state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], state["device"])


def _micro(model, tokens, scale):
    loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / scale
    loss.backward()
    return loss.detach()


def _agree_all(control, world_size, flag):
    if control is None:
        return bool(flag)
    try:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, bool(flag), group=control)
        return all(value is True for value in gathered)
    except Exception:
        return False


def build(context):
    state = context["state"]
    references = context.get("references") or {}
    parent_run = references.get(CALL_ID)

    def invoke_eager(state):
        if parent_run is not None:
            return parent_run(state)
        model = state["model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        n_micro = len(inputs)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i, tokens in enumerate(inputs):
            final = i + 1 == n_micro
            ctx = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
            with ctx:
                losses.append(_micro(model, tokens, n_micro))
        stacked = torch.stack(losses)
        total = stacked.sum()
        assert torch.isfinite(total)
        optimizer.step()
        if state["device"].type == "cuda":
            torch.cuda.synchronize(state["device"])
        return {"loss": total, "micro_losses": stacked}

    control = state.get("control")
    world_size = int(state.get("world_size", 1))
    capture_ok = False
    device = state.get("device")
    is_cuda = device is not None and device.type == "cuda"
    graph = None
    static_grads = None
    static_losses = None
    params = None
    model = None
    optimizer = None
    inputs = None
    n_micro = 0

    try:
        device = state["device"]
        if device.type != "cuda":
            raise RuntimeError("graph tier requires a cuda device")
        model = state["model"]
        raw_model = state["raw_model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        n_micro = len(inputs)
        if n_micro < 2:
            raise RuntimeError("workload must contain more than one micro-batch")
        params = list(raw_model.parameters())

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, tokens in enumerate(inputs):
                final = i + 1 == n_micro
                ctx = model.no_sync() if isinstance(model, DDP) and not final else nullcontext()
                with ctx:
                    losses.append(_micro(model, tokens, n_micro))
            total = torch.stack(losses).sum()
            assert torch.isfinite(total)
            optimizer.step()
            torch.cuda.synchronize(device)

        static_grads = [p.grad for p in params]
        if any(grad is None for grad in static_grads):
            raise RuntimeError("expected steady-state gradient views after priming")

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):
                torch._foreach_zero_(static_grads)
                for i in range(n_micro - 1):
                    with model.no_sync():
                        _micro(model, inputs[i], n_micro)
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)

        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        static_losses = []
        with torch.cuda.graph(graph, pool=pool, stream=side):
            torch._foreach_zero_(static_grads)
            for i in range(n_micro - 1):
                with model.no_sync():
                    static_losses.append(_micro(model, inputs[i], n_micro))
        torch.cuda.synchronize(device)
        torch._foreach_zero_(static_grads)
        torch.cuda.synchronize(device)
        capture_ok = True
    except Exception:
        capture_ok = False

    capture_ok = _agree_all(control, world_size, capture_ok)
    use_graph = False

    if capture_ok:
        try:
            def invoke_graph(state):
                for p, g in zip(params, static_grads):
                    p.grad = g
                graph.replay()
                losses = [t.clone() for t in static_losses]
                tokens = inputs[-1]
                with nullcontext():
                    final_loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / n_micro
                    final_loss.backward()
                    losses.append(final_loss.detach())
                stacked = torch.stack(losses)
                total = stacked.sum()
                assert torch.isfinite(total)
                optimizer.step()
                torch.cuda.synchronize(device)
                return {"loss": total, "micro_losses": stacked}

            _reset_like(state)
            result_graph = invoke_graph(state)
            grads_graph = [p.grad.detach().clone() for p in params]
            params_graph = [p.detach().clone() for p in params]
            _reset_like(state)
            result_eager = invoke_eager(state)
            grads_eager = [p.grad.detach().clone() for p in params]
            params_eager = [p.detach().clone() for p in params]
            ok = bool(torch.isfinite(result_graph["loss"]).item())
            ok = ok and torch.allclose(result_graph["loss"], result_eager["loss"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            ok = ok and torch.allclose(result_graph["micro_losses"], result_eager["micro_losses"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            for a, b in zip(grads_graph, grads_eager):
                ok = ok and torch.allclose(a, b, atol=GRAD_ATOL, rtol=GRAD_RTOL)
            for a, b in zip(params_graph, params_eager):
                ok = ok and torch.allclose(a, b, atol=PARAM_ATOL, rtol=PARAM_RTOL)
            use_graph = bool(ok)
        except Exception:
            use_graph = False

    use_graph = _agree_all(control, world_size, use_graph) if capture_ok else False
    _reset_like(state)
    if is_cuda:
        torch.cuda.synchronize()

    if use_graph:
        def invoke(state, _graph=invoke_graph, _state=state):
            if state is not _state:
                return invoke_eager(state)
            return _graph(state)

        return {CALL_ID: invoke}

    graph = None
    static_grads = None
    static_losses = None
    gc.collect()
    if is_cuda:
        torch.cuda.synchronize()
    return {CALL_ID: invoke_eager}
