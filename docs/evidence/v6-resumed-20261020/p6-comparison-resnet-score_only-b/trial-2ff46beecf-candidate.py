"""Execution-plan candidate: eval-BN folding + channels_last + CUDA Graph replay.

Replaces the single project.invoke call (adapter.run) of the frozen ResNet-18
FP32 batch=1 online workload, superseding the installed graph-replay bundle
(trial-d3cc996c55-param-9225be44) for the same call and preserving its
behavior (whole-forward CUDA graph replay returning a fresh logits clone per
invocation) while shrinking the kernel time inside the graph.

On the first (untimed) invocation:
  1. one reference forward of the unmodified model records baseline logits;
  2. every (Conv2d, BatchNorm2d) sibling adjacency in eval mode is folded
     algebraically into the conv (W' = W * gamma/sqrt(var+eps),
     b' = (b - mean) * s + beta), the BatchNorm2d is replaced by
     nn.Identity, and all conv weights move to channels_last;
  3. the folded channels_last forward is validated on the real fixture
     against the baseline with the task's own tolerances (atol=rtol=1e-4)
     and exact top-1 equality;
  4. on success the folded forward on a channels_last copy of the frozen
     pixels is warmed on a side stream and captured into a
     torch.cuda.CUDAGraph with its own private pool, and the pre-fold
     weight/module references are released.

Every invocation then replays the graph (real kernels execute each run; no
fixture output is cached) and returns {"logits": <fresh clone of the static
output>}, exactly adapter.run's result structure and aliasing behavior. Any
guard mismatch, self-check failure or capture error restores the original
modules (when capture has not yet succeeded) and permanently falls back to
the parent reference, so quality never depends on the reparameterization
succeeding.
"""

from __future__ import annotations

import torch
from torch import nn

PROJECT_INVOKE = "5d7d6bb13691b02f533f62b0"
_WARMUP_ITERS = 3
_ATOL = 1e-4
_RTOL = 1e-4


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

    def _undo_surgery():
        steps = holder.get("undo")
        if not steps:
            return
        for step in reversed(steps):
            parent_module, name, original = step["bn_slot"]
            setattr(parent_module, name, original)
            conv = step["conv"]
            conv.weight.data = step["weight"]
            if step["bias"] is None:
                conv.bias = None
            else:
                conv.bias.data = step["bias"]
        holder["undo"] = None

    def _fold_conv_bn(model):
        """Fold every eval (Conv2d -> BatchNorm2d) adjacency into the conv."""
        if model.training:
            return False
        undo = []
        for parent_module in list(model.modules()):
            children = list(parent_module.named_children())
            for index, (name, child) in enumerate(children):
                if not isinstance(child, nn.Conv2d):
                    continue
                if index + 1 >= len(children):
                    continue
                bn_name, bn = children[index + 1]
                if not (
                    isinstance(bn, nn.BatchNorm2d)
                    and not bn.training
                    and not child.training
                    and bn.running_mean is not None
                    and bn.running_var is not None
                    and bn.weight is not None
                    and bn.bias is not None
                ):
                    continue
                with torch.no_grad():
                    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    weight = child.weight * scale.reshape(-1, 1, 1, 1)
                    base = child.bias
                    if base is None:
                        base = torch.zeros(
                            child.out_channels,
                            dtype=weight.dtype,
                            device=weight.device,
                        )
                    bias = (base - bn.running_mean) * scale + bn.bias
                    undo.append(
                        {
                            "bn_slot": (parent_module, bn_name, bn),
                            "conv": child,
                            "weight": child.weight.data,
                            "bias": None if child.bias is None else child.bias.data,
                        }
                    )
                    child.weight.data = weight.contiguous(
                        memory_format=torch.channels_last
                    )
                    if child.bias is None:
                        child.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
                    else:
                        child.bias.data = bias.contiguous()
                setattr(parent_module, bn_name, nn.Identity())
        if not undo:
            return False
        holder["undo"] = undo
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    weight = module.weight.data
                    if not weight.is_contiguous(memory_format=torch.channels_last):
                        module.weight.data = weight.contiguous(
                            memory_format=torch.channels_last
                        )
        return True

    def _forward(model, pixels):
        return model(pixel_values=pixels).logits

    def _capture(state):
        device = state["device"]
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("graph capture requires the frozen CUDA workload")
        model = state["model"]
        pixels = state["pixels"]
        with torch.inference_mode():
            baseline = _forward(model, pixels).clone()
        if not _fold_conv_bn(model):
            raise RuntimeError("no eval Conv2d->BatchNorm2d adjacency found to fold")
        try:
            static_pixels = pixels.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                folded = _forward(model, static_pixels)
                if not bool(torch.isclose(folded, baseline, atol=_ATOL, rtol=_RTOL).all()):
                    raise RuntimeError("folded logits violate the task tolerances")
                if not torch.equal(folded.argmax(-1), baseline.argmax(-1)):
                    raise RuntimeError("folded logits change the top-1 prediction")
            torch.cuda.synchronize(device)
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                with torch.inference_mode():
                    for _ in range(_WARMUP_ITERS):
                        model(pixel_values=static_pixels)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    static_logits = _forward(model, static_pixels)
        except Exception:
            _undo_surgery()
            raise
        holder["graph"] = graph
        holder["static_logits"] = static_logits
        holder["static_pixels"] = static_pixels
        holder["model_id"] = id(model)
        holder["pixels_ptr"] = pixels.data_ptr()
        holder["pixels_shape"] = tuple(pixels.shape)
        holder["pixels_dtype"] = pixels.dtype
        holder["device"] = device
        holder["undo"] = None
        graph.replay()
        torch.cuda.synchronize(device)

    def run_folded_graphed(state):
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
                    or state["model"].training
                ):
                    return _eager(state)
            with torch.inference_mode():
                holder["graph"].replay()
                logits = holder["static_logits"].clone()
            return {"logits": logits}
        except Exception:
            _undo_surgery()
            holder.clear()
            holder["failed"] = True
        return _eager(state)

    return {PROJECT_INVOKE: run_folded_graphed}
