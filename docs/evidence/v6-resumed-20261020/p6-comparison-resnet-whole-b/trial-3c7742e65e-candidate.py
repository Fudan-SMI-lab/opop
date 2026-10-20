from __future__ import annotations

import torch

PROJECT_INVOKE_CALL = "5d7d6bb13691b02f533f62b0"

_SIDE_STREAM_WARMUP_ITERS = 3
_EXPECTED_INPUT_SHAPE = (1, 3, 224, 224)
_ATOL = 1e-4
_RTOL = 1e-4


def build(context):
    references = context.get("references") or {}
    parent_run = references.get(PROJECT_INVOKE_CALL)

    plan = {
        "broken": False,
        "graph": None,
        "static_input": None,
        "static_logits": None,
        "model": None,
    }

    def _parent(state):
        if parent_run is not None:
            return parent_run(state)
        with torch.inference_mode():
            return {"logits": state["model"](pixel_values=state["pixels"]).logits}

    def _input_ok(pixels):
        return (
            pixels.is_cuda
            and pixels.dim() == 4
            and tuple(pixels.shape) == _EXPECTED_INPUT_SHAPE
            and pixels.dtype == torch.float32
        )

    def _drop_graph():
        plan["graph"] = None
        plan["static_input"] = None
        plan["static_logits"] = None
        plan["model"] = None

    def _capture(state):
        model = state["model"]
        pixels = state["pixels"]
        torch.cuda.synchronize()
        static_input = torch.empty_like(pixels)
        static_input.copy_(pixels)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_SIDE_STREAM_WARMUP_ITERS):
                eager_logits = model(pixel_values=static_input).logits
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = model(pixel_values=static_input).logits
        graph.replay()
        torch.cuda.synchronize()
        if not torch.allclose(static_logits, eager_logits, atol=_ATOL, rtol=_RTOL):
            raise RuntimeError("replayed logits diverge from eager logits")
        plan["graph"] = graph
        plan["static_input"] = static_input
        plan["static_logits"] = static_logits
        plan["model"] = model

    @torch.inference_mode()
    def _invoke(state):
        try:
            if plan["broken"]:
                return _parent(state)
            device = state.get("device")
            if device is None or device.type != "cuda" or not torch.cuda.is_available():
                plan["broken"] = True
                return _parent(state)
            if not _input_ok(state["pixels"]):
                return _parent(state)
            if plan["graph"] is not None and state["model"] is not plan["model"]:
                _drop_graph()
            if plan["graph"] is None:
                try:
                    _capture(state)
                except Exception:
                    _drop_graph()
                    plan["broken"] = True
                    return _parent(state)
            plan["static_input"].copy_(state["pixels"])
            plan["graph"].replay()
            return {"logits": plan["static_logits"].clone()}
        except Exception:
            _drop_graph()
            plan["broken"] = True
            return _parent(state)

    return {PROJECT_INVOKE_CALL: _invoke}
