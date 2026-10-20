"""Execution-plan candidate: CUDA-graph replay of the NCCL-free no_sync
accumulation micro-steps of the GPT-2 FP32 DDP training workload.

Only the single project.invoke call is replaced. The three no_sync
forward/backward micro-batches (no collectives: DDP reducer hooks are inert
under no_sync) are captured once during build into one CUDA graph and replayed
per repetition; the final accumulation micro-batch with its 13 real gradient
allreduces, the loss stack/sum + isfinite boundary and exactly one SGD update
stay eager on the native C++ reducer path. The per-parameter SGD update can
additionally be replayed from a second CUDA graph. No inner call site is
wrapped or intercepted.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # pragma: no cover

    @contextmanager
    def _phase(name):
        yield


PROJECT_CALL = "5d7d6bb13691b02f533f62b0"
ROUTES = ("full", "accum", "eager")
_PLANS = {}


def _restore(state):
    state["raw_model"].load_state_dict(state["initial"])
    state["optimizer"].load_state_dict(state["initial_optimizer_state"])
    state["optimizer"].zero_grad(set_to_none=True)
    torch.set_rng_state(state["cpu_rng_state"])
    if state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], state["device"])


def _eager_step(state):
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    n = len(inputs)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for i, tokens in enumerate(inputs):
        is_final = i + 1 == n
        context = model.no_sync() if hasattr(model, "no_sync") and not is_final else nullcontext()
        with context, _phase("accumulate_micro_step_final_allreduce" if is_final else "accumulate_micro_step_local"):
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / n
            loss.backward()
            losses.append(loss.detach())
    micro_losses = torch.stack(losses)
    loss = micro_losses.sum()
    assert torch.isfinite(loss)
    with _phase("optimizer_step"):
        optimizer.step()
    if state["device"].type == "cuda":
        torch.cuda.synchronize(state["device"])
    return {"loss": loss, "micro_losses": micro_losses}


def _local_micros(model, inputs, n, losses_out):
    with model.no_sync():
        for i in range(n - 1):
            tokens = inputs[i]
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / n
            loss.backward()
            losses_out.append(loss.detach())


def _graph_step(state, core, use_opt_graph):
    model = core["model"]
    optimizer = core["optimizer"]
    inputs = core["inputs"]
    n = core["n"]
    for parameter, static_grad in core["grad_pairs"]:
        parameter.grad = static_grad
    core["accum_graph"].replay()
    with _phase("accumulate_micro_step_final_allreduce"):
        loss_final = model(input_ids=inputs[n - 1], labels=inputs[n - 1], use_cache=False).loss / n
        loss_final.backward()
    losses = list(core["static_losses"]) + [loss_final.detach()]
    micro_losses = torch.stack(losses)
    loss = micro_losses.sum()
    assert torch.isfinite(loss)
    with _phase("optimizer_step"):
        if use_opt_graph and core["opt_graph"] is not None:
            core["opt_graph"].replay()
        else:
            optimizer.step()
    if core["device"].type == "cuda":
        torch.cuda.synchronize(core["device"])
    return {"loss": loss, "micro_losses": micro_losses}


def _snapshot(state, result):
    raw_model = state["raw_model"]
    return {
        "loss": result["loss"].detach().to("cpu", copy=True),
        "micro_losses": result["micro_losses"].detach().to("cpu", copy=True),
        "gradients": {name: parameter.grad.detach().to("cpu", copy=True) for name, parameter in raw_model.named_parameters()},
        "parameters": {name: parameter.detach().to("cpu", copy=True) for name, parameter in raw_model.named_parameters()},
    }


def _same(a, b):
    try:
        if set(a) != set(b):
            return False
        if not torch.allclose(a["loss"], b["loss"], atol=1e-06, rtol=1e-04, equal_nan=False):
            return False
        if not torch.allclose(a["micro_losses"], b["micro_losses"], atol=1e-06, rtol=1e-04, equal_nan=False):
            return False
        if set(a["gradients"]) != set(b["gradients"]) or set(a["parameters"]) != set(b["parameters"]):
            return False
        for name in a["gradients"]:
            if not torch.allclose(a["gradients"][name], b["gradients"][name], atol=1e-06, rtol=1e-04, equal_nan=False):
                return False
        for name in a["parameters"]:
            if not torch.allclose(a["parameters"][name], b["parameters"][name], atol=1e-06, rtol=1e-04, equal_nan=False):
                return False
        return True
    except Exception:
        return False


def _exchange(control, ok):
    try:
        if not (dist.is_available() and dist.is_initialized()):
            return bool(ok)
        flag = torch.tensor([0 if ok else 1], dtype=torch.int64)
        if control is not None:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=control)
        else:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() == 0
    except Exception:
        return False


def _prepare(state, route):
    diag = {}
    control = state.get("control")
    model = state["model"]
    optimizer = state["optimizer"]
    inputs = state["inputs"]
    raw_model = state["raw_model"]
    device = state["device"]
    n = len(inputs)
    params = list(raw_model.parameters())

    try:
        ok = (
            device.type == "cuda"
            and torch.cuda.is_available()
            and n >= 2
            and hasattr(model, "no_sync")
            and all(torch.is_tensor(t) and t.is_cuda for t in inputs)
            and all(p.is_cuda for p in params)
        )
    except Exception:
        ok = False
    if not _exchange(control, ok):
        return None, "eager", diag

    stable_ptr = None
    try:
        _eager_step(state)
        first = [p.grad.data_ptr() for p in params]
        _eager_step(state)
        second = [p.grad.data_ptr() for p in params]
        if first != second:
            _eager_step(state)
            second = [p.grad.data_ptr() for p in params]
        stable_ptr = second
        ok = all(p.grad is not None for p in params)
    except Exception as exc:
        ok = False
        diag["settle"] = repr(exc)[:300]
    if not _exchange(control, ok):
        return None, "eager", diag

    accum_graph = None
    static_losses = []
    static_grads = []
    try:
        _restore(state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                _local_micros(model, inputs, n, losses)
        torch.cuda.current_stream().wait_stream(side)
        optimizer.zero_grad(set_to_none=True)
        accum_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(accum_graph):
            _local_micros(model, inputs, n, static_losses)
        static_grads = [p.grad for p in params]
        ok = len(static_losses) == n - 1 and all(g is not None for g in static_grads)
    except Exception as exc:
        ok = False
        accum_graph = None
        static_losses = []
        static_grads = []
        diag["capture_accum"] = repr(exc)[:300]
    if not _exchange(control, ok):
        return None, "eager", diag

    opt_graph = None
    try:
        for parameter, static_grad in zip(params, static_grads):
            parameter.grad = static_grad
        accum_graph.replay()
        loss_final = model(input_ids=inputs[n - 1], labels=inputs[n - 1], use_cache=False).loss / n
        loss_final.backward()
        view_ptrs = [p.grad.data_ptr() for p in params]
        if route == "full" and view_ptrs == stable_ptr:
            opt_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(opt_graph):
                optimizer.step()
        ok = True
    except Exception as exc:
        ok = False
        opt_graph = None
        diag["capture_opt"] = repr(exc)[:300]
    if not _exchange(control, ok):
        return None, "eager", diag
    if not _exchange(control, route == "full" and opt_graph is not None):
        opt_graph = None

    core = {
        "model": model,
        "optimizer": optimizer,
        "inputs": inputs,
        "n": n,
        "params": params,
        "device": device,
        "accum_graph": accum_graph,
        "static_losses": static_losses,
        "grad_pairs": list(zip(params, static_grads)),
        "opt_graph": opt_graph,
    }

    ok_accum = False
    ok_full = False
    try:
        _restore(state)
        refs = [_snapshot(state, _eager_step(state))]
        refs.append(_snapshot(state, _eager_step(state)))
        _restore(state)
        got = []
        for _ in range(2):
            got.append(_snapshot(state, _graph_step(state, core, False)))
        ok_accum = all(_same(refs[i], got[i]) for i in range(2))
        del got
        if opt_graph is not None:
            _restore(state)
            got2 = []
            for _ in range(2):
                got2.append(_snapshot(state, _graph_step(state, core, True)))
            ok_full = all(_same(refs[i], got2[i]) for i in range(2))
            del got2
    except Exception as exc:
        ok_accum = False
        ok_full = False
        diag["verify"] = repr(exc)[:300]
    if not _exchange(control, ok_accum):
        return None, "eager", diag
    if opt_graph is not None:
        if not _exchange(control, ok_full):
            core["opt_graph"] = None
            effective = "accum"
        else:
            effective = "full"
    else:
        effective = "accum"
    _restore(state)
    return core, effective, diag


def _make_replacement(core, effective, state, parent, record_route):
    use_opt = effective == "full"
    label = "graph_full" if use_opt else "graph_accum"

    def _replacement(state_arg):
        if state_arg is not state:
            return parent(state_arg)
        try:
            if record_route is not None:
                record_route(PROJECT_CALL, label)
        except Exception:
            pass
        return _graph_step(state_arg, core, use_opt)

    return _replacement


def build(context):
    state = context["state"]
    parameters = dict(context.get("parameters") or {})
    route = parameters.get("route", "full")
    if route not in ROUTES:
        route = "full"
    parent = context["references"][PROJECT_CALL]
    record_route = context.get("record_route")
    key = (id(state), route)
    cached = _PLANS.get(key)
    if cached is not None:
        return {PROJECT_CALL: cached}
    replacement = None
    if route != "eager":
        try:
            core, effective, _diag = _prepare(state, route)
        except Exception:
            core, effective = None, "eager"
        if core is not None:
            replacement = _make_replacement(core, effective, state, parent, record_route)
    if replacement is None:

        def _parent_route(state_arg):
            try:
                if record_route is not None:
                    record_route(PROJECT_CALL, "parent")
            except Exception:
                pass
            return parent(state_arg)

        replacement = _parent_route
    _PLANS[key] = replacement
    return {PROJECT_CALL: replacement}
