"""Execution-plan candidate: whole-forward CUDA Graph capture + replay.

Replaces the single project.invoke call (adapter.run) for the frozen ResNet-18
FP32 batch=1 online workload. On the first (untimed) invocation the fixed
forward ``model(pixel_values=state["pixels"]).logits`` under
``torch.inference_mode()`` is warmed on a side stream and captured into a
``torch.cuda.CUDAGraph`` with its own private memory pool; every invocation
then replays the graph and returns ``{"logits": <fresh clone of the static
output>}``, which is exactly adapter.run's result structure and aliasing
behavior (a new logits tensor per run). Replay executes the real kernels each
invocation; no fixture output is cached.

The replay path is guarded by the frozen static-input invariants (model object
identity, pixels data_ptr/shape/dtype, device). Any guard mismatch or capture
failure permanently falls back to the unmodified parent (adapter.run via
context["references"]), so quality never depends on graph capture succeeding.
"""

from __future__ import annotations

import torch

PROJECT_INVOKE = "5d7d6bb13691b02f533f62b0"
_WARMUP_ITERS = 3


def build(context):
    references = context.get("references") or {}
    targets = context.get("targets") or {}
    if PROJECT_INVOKE not in targets:
        return {}

    def _eager(state):
        parent = references.get(PROJECT_INVOKE)
        if callable(parent):
            return parent(state)
        with torch.inference_mode():
            logits = state["model"](pixel_values=state["pixels"]).logits
        return {"logits": logits}

    holder = {}

    def _capture(state):
        device = state["device"]
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("graph capture requires the frozen CUDA workload")
        model = state["model"]
        pixels = state["pixels"]
        torch.cuda.synchronize(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            with torch.inference_mode():
                for _ in range(_WARMUP_ITERS):
                    model(pixel_values=pixels)
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                static_logits = model(pixel_values=pixels).logits
        holder["graph"] = graph
        holder["static_logits"] = static_logits
        holder["model_id"] = id(model)
        holder["pixels_ptr"] = pixels.data_ptr()
        holder["pixels_shape"] = tuple(pixels.shape)
        holder["pixels_dtype"] = pixels.dtype
        holder["device"] = device
        graph.replay()
        torch.cuda.synchronize(device)

    def run_graphed(state):
        if holder.get("failed"):
            return _eager(state)
        try:
            if "graph" not in holder:
                _capture(state)
            else:
                pixels = state["pixels"]
                if (
                    id(state["model"]) != holder["model_id"]
                    or pixels.data_ptr() != holder["pixels_ptr"]
                    or tuple(pixels.shape) != holder["pixels_shape"]
                    or pixels.dtype != holder["pixels_dtype"]
                    or state["device"] != holder["device"]
                ):
                    return _eager(state)
            with torch.inference_mode():
                holder["graph"].replay()
                logits = holder["static_logits"].clone()
            return {"logits": logits}
        except Exception:
            holder.clear()
            holder["failed"] = True
        return _eager(state)

    return {PROJECT_INVOKE: run_graphed}
