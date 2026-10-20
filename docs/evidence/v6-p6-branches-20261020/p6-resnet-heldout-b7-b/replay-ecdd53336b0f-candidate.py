"""Execution-plan candidate: whole-forward CUDA graph delivery for the frozen
batch=1 FP32 ResNet-18 online inference workload.

Selected sites (groups 3c57c8731ec33ed76cbb, 6870b4a6e1536342a2ef):
  5d7d6bb13691b02f533f62b0  project.invoke (adapter.run boundary)
  80ff8693d7ac3cf2433b10f2  ResNetForImageClassification forward

At build time, outside the timed region, the unchanged parent forward is
warmed on a side stream and captured once into static input/output buffers.
Per call the adapter's pixels are copied into the static input, the graph is
replayed (the identical captured kernel DAG executes on the current values),
and the static logits are cloned, so the workload executes on every
invocation. Any signature mismatch or build-time/runtime failure falls back
atomically to the parent bound forward / adapter.run.
"""
from __future__ import annotations

import torch

PROJECT_CALL = "5d7d6bb13691b02f533f62b0"
MODEL_CALL = "80ff8693d7ac3cf2433b10f2"

_EXPECTED_SHAPE = (1, 3, 224, 224)
_WARMUP_ITERS = 3
_VERIFY_ATOL = 5e-6
_VERIFY_RTOL = 5e-6


def _sig_ok(tensor, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == _EXPECTED_SHAPE
        and tensor.dtype == torch.float32
        and tensor.is_cuda
        and tensor.device == device
        and tensor.is_contiguous()
        and not tensor.requires_grad
    )


def build(context):
    references = dict(context.get("references") or {})
    parent_project = references[PROJECT_CALL]
    parent_model = references[MODEL_CALL]

    def _project_passthrough(state):
        return parent_project(state)

    def _model_passthrough(*args, **kwargs):
        return parent_model(*args, **kwargs)

    try:
        crafted = _craft(context, references, parent_project, parent_model)
    except Exception:
        crafted = None
    if crafted is None:
        return {PROJECT_CALL: _project_passthrough, MODEL_CALL: _model_passthrough}
    return crafted


def _craft(context, references, parent_project, parent_model):
    if not (hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph")):
        return None
    if not torch.cuda.is_available():
        return None
    state = context.get("state") or {}
    device = state.get("device")
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    model = (context.get("models") or {}).get("model") or state.get("model")
    pixels = state.get("pixels")
    if not isinstance(model, torch.nn.Module) or not _sig_ok(pixels, device):
        return None

    try:
        from transformers.modeling_outputs import ImageClassifierOutput

        def make_output(logits):
            return ImageClassifierOutput(loss=None, logits=logits)

    except Exception:

        class _OutputShim:
            def __init__(self, logits):
                self.logits = logits

        def make_output(logits):
            return _OutputShim(logits)

    with torch.cuda.device(device):
        with torch.inference_mode(False):
            static_input = torch.empty_like(pixels)
        static_input.copy_(pixels)

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(_WARMUP_ITERS):
                model(pixel_values=static_input)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_out = model(pixel_values=static_input)
        static_logits = static_out.logits
        if not isinstance(static_logits, torch.Tensor):
            return None
        torch.cuda.synchronize(device)

        with torch.inference_mode():
            def _replay_read(source):
                static_input.copy_(source)
                graph.replay()
                torch.cuda.synchronize(device)
                return static_logits.clone()

            def _eager_read(source):
                return model(pixel_values=source).logits

            replay_a = _replay_read(pixels)
            eager_a = _eager_read(pixels)
            probe = torch.flip(pixels, dims=[-1]).contiguous()
            replay_b = _replay_read(probe)
            eager_b = _eager_read(probe)
            static_input.copy_(pixels)
            torch.cuda.synchronize(device)

        for replayed, reference in ((replay_a, eager_a), (replay_b, eager_b)):
            if not torch.allclose(
                replayed.float(), reference.float(), atol=_VERIFY_ATOL, rtol=_VERIFY_RTOL
            ):
                return None
            if not torch.equal(replayed.argmax(-1), reference.argmax(-1)):
                return None

        del replay_a, eager_a, replay_b, eager_b, probe
        torch.cuda.empty_cache()

    flags = {"model_calls": 0, "inline": False}

    def model_forward(pixel_values=None, *args, **kwargs):
        flags["model_calls"] += 1
        if args or kwargs or not _sig_ok(pixel_values, device):
            return parent_model(pixel_values, *args, **kwargs)
        try:
            with torch.inference_mode():
                static_input.copy_(pixel_values)
                graph.replay()
                logits = static_logits.clone()
        except Exception:
            return parent_model(pixel_values)
        return make_output(logits)

    def inline_replay(source):
        with torch.inference_mode():
            static_input.copy_(source)
            graph.replay()
            return {"logits": static_logits.clone()}

    def project_invoke(state):
        if not isinstance(state, dict):
            return parent_project(state)
        pixels = state.get("pixels")
        model = state.get("model")
        if not _sig_ok(pixels, device):
            return parent_project(state)
        if flags["inline"] or not isinstance(model, torch.nn.Module):
            try:
                return inline_replay(pixels)
            except Exception:
                return parent_project(state)
        before = flags["model_calls"]
        try:
            with torch.inference_mode():
                out = model(pixel_values=pixels)
        except Exception:
            return parent_project(state)
        logits = getattr(out, "logits", None)
        routed = flags["model_calls"] != before and isinstance(logits, torch.Tensor)
        if not routed:
            flags["inline"] = True
        if isinstance(logits, torch.Tensor):
            return {"logits": logits}
        return parent_project(state)

    return {PROJECT_CALL: project_invoke, MODEL_CALL: model_forward}
