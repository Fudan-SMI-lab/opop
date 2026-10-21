"""Whole-bundle execution-policy replacement for the complete-workload project
call 5d7d6bb13691b02f533f62b0 (group 3c57c8731ec33ed76cbb).

Repair of the two rejected trials: trial-5a705e709d rerouted the 13 bucket
allreduces onto a second NCCL process group with forced NCCL env tuning and
lost the step constraint; trial-43f18a5360 bound the two allreduce_ sites
functionally under the fixed legacy backend (binding_cost-2087a4f7 measures
+92.3/+100.8 ms of clean step time from that additive dispatch alone) and its
gloo barrier propagated release skew into the burst. This candidate therefore
declares zero functional sites: a single direct project.invoke boundary whose
replacement re-implements adapter.run with identical semantics - zero_grad
inside the timed boundary, loss scaling by len(inputs), no_sync accumulation
on the first accumulation_steps-1 micro-batches, the synchronized DDP
all-reduce on the final backward, one SGD step, one final
torch.cuda.synchronize - while executing those no_sync micro-batches through
the underlying module as captured CUDA graph replays of the parent's own
work. Per invocation the static input buffers are refreshed from the current
inputs and gradients accumulate in place into persistent per-parameter
buffers that are primed and zeroed before each replay; the final synchronized
micro-batch then runs eagerly through the unmodified DDP wrapper, whose
reducer sees exactly the same defined non-bucket-view gradients as in the
parent (where autograd allocates them) and copies them into its buckets
before the unchanged 13-bucket all-reduce on the default NCCL process group.
CPU verification on two-rank gloo DDP confirms this chain bit-exact against
both the parent step and a serial no-DDP reference. The first invocation runs
the pure parent reference (bucket rebuild, library warmup, exact outputs);
the second (untimed warmup) allocates the buffers and captures one graph per
no_sync micro-batch into a shared pool; later invocations replay. Any capture
failure, out-of-place accumulation flavor or workload-shape change disables
the graph path permanently and routes the whole invocation through the parent
reference with a distinct label; the collective-bearing final micro-batch and
optimizer step are never re-run by a fallback.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase
except ImportError:
    def phase(name):
        return nullcontext()

KNOWN_PROJECT_CALL = "5d7d6bb13691b02f533f62b0"

_LABEL_FIRST = "eager:parent:first_invocation"
_LABEL_CAPTURE = "capture:no_sync_graphs+eager_final_micro"
_LABEL_REPLAY = "replay:no_sync_graphs+eager_final_micro"


def _project_call_id(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    if KNOWN_PROJECT_CALL in targets and KNOWN_PROJECT_CALL in references:
        return KNOWN_PROJECT_CALL
    for call_id, spec in targets.items():
        if (
            isinstance(spec, dict)
            and spec.get("kind") == "project"
            and spec.get("target") == "project.invoke"
            and call_id in references
        ):
            return call_id
    return None


def build(context):
    context = context or {}
    record = context.get("record_route") or (lambda *_: None)
    references = context.get("references") or {}
    state = context.get("state") or {}

    call_id = _project_call_id(context)
    reference = references.get(call_id) if call_id is not None else None
    if call_id is None or reference is None:
        return {}

    model = state.get("model")
    raw_model = state.get("raw_model") or model
    inputs = state.get("inputs")
    if (
        not isinstance(model, DDP)
        or not isinstance(raw_model, torch.nn.Module)
        or state.get("optimizer") is None
        or not isinstance(inputs, (list, tuple))
        or len(inputs) < 2
        or not torch.cuda.is_available()
    ):
        return {}

    book = {
        "calls": 0,
        "failed": None,
        "model": None,
        "raw_model": None,
        "params": None,
        "buffers": None,
        "graphs": None,
        "static_inputs": None,
        "static_losses": None,
    }

    def _fallback(state, label):
        record(call_id, label)
        return reference(state)

    def _prime_and_clear(params, buffers):
        for parameter, buffer in zip(params, buffers):
            parameter.grad = buffer
        try:
            torch._foreach_zero_(buffers)
        except Exception:
            for buffer in buffers:
                buffer.zero_()

    def _check_shapes(inputs):
        graphs = book["graphs"]
        static_inputs = book["static_inputs"]
        if len(inputs) - 1 != len(graphs):
            return False
        for index in range(len(graphs)):
            if (
                inputs[index].shape != static_inputs[index].shape
                or inputs[index].dtype != static_inputs[index].dtype
                or inputs[index].device != static_inputs[index].device
            ):
                return False
        return True

    def _run_segments(inputs):
        graphs = book["graphs"]
        static_inputs = book["static_inputs"]
        static_losses = book["static_losses"]
        losses = []
        for index, graph in enumerate(graphs):
            static_inputs[index].copy_(inputs[index], non_blocking=True)
            graph.replay()
            losses.append(static_losses[index].detach().clone())
        return losses

    def _capture_segments(raw_model, optimizer, inputs):
        params = book["params"]
        if book["buffers"] is None:
            book["buffers"] = [torch.zeros_like(parameter) for parameter in params]
        buffers = book["buffers"]
        segments = len(inputs) - 1
        optimizer.zero_grad(set_to_none=True)
        _prime_and_clear(params, buffers)
        static_inputs = [inputs[index].detach().clone() for index in range(segments)]
        static_losses = []
        graphs = []
        pool = torch.cuda.graph_pool_handle()
        for index in range(segments):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = raw_model(
                    input_ids=static_inputs[index],
                    labels=static_inputs[index],
                    use_cache=False,
                )
                segment_loss = output.loss / len(inputs)
                segment_loss.backward()
            static_losses.append(segment_loss)
            graphs.append(graph)
        for parameter, buffer in zip(params, buffers):
            if parameter.grad is not buffer:
                raise RuntimeError("out-of-place gradient accumulation detected")
        book["graphs"] = graphs
        book["static_inputs"] = static_inputs
        book["static_losses"] = static_losses
        return _run_segments(inputs)

    def _replay_segments(optimizer, inputs):
        if not _check_shapes(inputs):
            raise RuntimeError("workload_shape_changed")
        optimizer.zero_grad(set_to_none=True)
        _prime_and_clear(book["params"], book["buffers"])
        return _run_segments(inputs)

    def _eager_final_micro(model, inputs):
        tokens = inputs[-1]
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
        loss.backward()
        return loss.detach()

    def replacement(state):
        book["calls"] += 1
        if book["failed"] is not None:
            return _fallback(state, "fallback:parent:steady_state:" + book["failed"])

        model = state.get("model")
        raw_model = state.get("raw_model") or model
        optimizer = state.get("optimizer")
        inputs = state.get("inputs")
        if (
            not isinstance(model, DDP)
            or not isinstance(raw_model, torch.nn.Module)
            or optimizer is None
            or not isinstance(inputs, (list, tuple))
            or len(inputs) < 2
        ):
            return _fallback(state, "fallback:parent:unsupported_signature")

        if book["params"] is None:
            result = reference(state)
            try:
                book["model"] = model
                book["raw_model"] = raw_model
                book["params"] = list(raw_model.parameters())
            except Exception:
                book["failed"] = "snapshot_failed"
            record(call_id, _LABEL_FIRST)
            return result

        if model is not book["model"] or raw_model is not book["raw_model"]:
            return _fallback(state, "fallback:parent:model_replaced")

        if book["graphs"] is not None:
            failure = None
            with phase("gradient_accumulation"):
                try:
                    losses = _replay_segments(optimizer, inputs)
                except Exception as exc:
                    book["failed"] = "replay_failed:" + type(exc).__name__
                    failure = "fallback:parent:" + book["failed"]
                if failure is None:
                    losses.append(_eager_final_micro(model, inputs))
                    loss = torch.stack(losses).sum()
            if failure is not None:
                return _fallback(state, failure)
            with phase("optimizer_step"):
                optimizer.step()
                torch.cuda.synchronize()
            record(call_id, _LABEL_REPLAY)
            return {"loss": loss.detach(), "micro_losses": losses}

        failure = None
        with phase("gradient_accumulation"):
            try:
                losses = _capture_segments(raw_model, optimizer, inputs)
            except Exception as exc:
                import traceback
                print("DEVELOPER_CAPTURE_STACK", traceback.format_exc(), flush=True)
                book["graphs"] = None
                book["static_inputs"] = None
                book["static_losses"] = None
                book["buffers"] = None
                book["failed"] = "capture_failed:" + type(exc).__name__
                failure = "fallback:parent:" + book["failed"]
            if failure is None:
                losses.append(_eager_final_micro(model, inputs))
                loss = torch.stack(losses).sum()
        if failure is not None:
            return _fallback(state, failure)
        with phase("optimizer_step"):
            optimizer.step()
            torch.cuda.synchronize()
        record(call_id, _LABEL_CAPTURE)
        return {"loss": loss.detach(), "micro_losses": losses}

    return {call_id: replacement}
