from __future__ import annotations

import torch

PROJECT_INVOKE_CALL = "5d7d6bb13691b02f533f62b0"

_ATOL = 1e-4
_RTOL = 1e-4
_EXPECTED_SHAPE = (1, 3, 224, 224)
_WARMUP_ITERS = 3
_TIMING_REPLAYS = 40
_CL_MARGIN_MS = 0.005
_PARKED_BN_ATTR = "_opop_parked_normalization"


class _FoldedConv2d(torch.nn.Module):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        return torch.nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def build(context):
    references = context.get("references") or {}
    parent_run = references.get(PROJECT_INVOKE_CALL)

    plan = {
        "broken": False,
        "graph": None,
        "static_logits": None,
        "model": None,
        "pixels": None,
        "pixels_token": None,
    }

    def _parent(state):
        if parent_run is not None:
            return parent_run(state)
        with torch.inference_mode():
            return {"logits": state["model"](pixel_values=state["pixels"]).logits}

    def _input_ok(pixels):
        return (
            isinstance(pixels, torch.Tensor)
            and pixels.is_cuda
            and pixels.dim() == 4
            and tuple(pixels.shape) == _EXPECTED_SHAPE
            and pixels.dtype == torch.float32
        )

    def _pixels_token(pixels):
        return (pixels.data_ptr(), tuple(pixels.shape), tuple(pixels.stride()), pixels.dtype, pixels.get_device())

    def _drop_graph():
        plan["graph"] = None
        plan["static_logits"] = None
        plan["model"] = None
        plan["pixels"] = None
        plan["pixels_token"] = None

    def _forward_logits(model, pixels):
        return model(pixel_values=pixels).logits

    def _side_stream_warmup(model, pixels):
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            logits = None
            for _ in range(_WARMUP_ITERS):
                logits = _forward_logits(model, pixels)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        return logits

    def _capture(model, pixels):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = _forward_logits(model, pixels)
        graph.replay()
        torch.cuda.synchronize()
        return graph, static_logits

    def _matches(out, ref):
        return bool(
            torch.allclose(out, ref, atol=_ATOL, rtol=_RTOL)
            and torch.equal(out.argmax(-1), ref.argmax(-1))
        )

    def _time_replay(graph):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        graph.replay()
        torch.cuda.synchronize()
        start.record()
        for _ in range(_TIMING_REPLAYS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / _TIMING_REPLAYS

    def _fold_batchnorms(model):
        pairs = []
        for module in model.modules():
            conv = getattr(module, "convolution", None)
            bn = getattr(module, "normalization", None)
            if not isinstance(conv, torch.nn.Conv2d):
                continue
            if not isinstance(bn, torch.nn.modules.batchnorm._BatchNorm):
                continue
            if (
                conv.training
                or bn.training
                or bn.running_mean is None
                or bn.running_var is None
                or bn.weight is None
                or bn.bias is None
            ):
                return None
            pairs.append((module, conv, bn))
        backups = []
        for module, conv, bn in pairs:
            with torch.no_grad():
                scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
                new_weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
                base_bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean)
                new_bias = base_bias * scale + bn.bias.detach() - bn.running_mean.detach() * scale
                backups.append(
                    {
                        "module": module,
                        "conv": conv,
                        "bn": bn,
                        "weight_backup": conv.weight.detach().clone(),
                        "bias_backup": conv.bias.detach().clone() if conv.bias is not None else None,
                    }
                )
                conv.weight.copy_(new_weight)
            module.convolution = _FoldedConv2d(
                conv.weight,
                new_bias.detach(),
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
            setattr(module, _PARKED_BN_ATTR, bn)
            module.normalization = torch.nn.Identity()
        return backups

    def _unfold_batchnorms(backups):
        for entry in reversed(backups):
            module = entry["module"]
            with torch.no_grad():
                entry["conv"].weight.copy_(entry["weight_backup"])
                if entry["bias_backup"] is not None:
                    entry["conv"].bias.copy_(entry["bias_backup"])
            module.convolution = entry["conv"]
            module.normalization = entry["bn"]
            delattr(module, _PARKED_BN_ATTR)

    def _setup(state):
        model = state["model"]
        pixels = state["pixels"]
        eager_ref = _side_stream_warmup(model, pixels).clone()

        backups = _fold_batchnorms(model)
        if backups:
            folded_eager = _forward_logits(model, pixels)
            if _matches(folded_eager, eager_ref):
                backups = None
            else:
                _unfold_batchnorms(backups)
                backups = None

        graph_a = None
        logits_a = None
        t_a = None
        try:
            graph_a, logits_a = _capture(model, pixels)
        except Exception:
            graph_a, logits_a = None, None
        if graph_a is not None and not _matches(logits_a, eager_ref):
            graph_a, logits_a = None, None
        if graph_a is not None:
            t_a = _time_replay(graph_a)

        graph_b = None
        logits_b = None
        t_b = None
        keep_cl = False
        params4d = [p for p in model.parameters() if p.dim() == 4]
        saved_data = [p.data for p in params4d]
        converted = False
        try:
            model.to(memory_format=torch.channels_last)
            converted = True
            _side_stream_warmup(model, pixels)
            cl_eager = _forward_logits(model, pixels)
            if _matches(cl_eager, eager_ref):
                graph_b, logits_b = _capture(model, pixels)
                if graph_b is not None and not _matches(logits_b, eager_ref):
                    graph_b, logits_b = None, None
                if graph_b is not None:
                    t_b = _time_replay(graph_b)
        except Exception:
            graph_b, logits_b = None, None

        if graph_b is not None and (graph_a is None or t_b < t_a - _CL_MARGIN_MS):
            keep_cl = True
        if not keep_cl and converted:
            for p, d in zip(params4d, saved_data):
                p.data = d
            graph_b, logits_b = None, None

        winner_graph = graph_b if keep_cl else graph_a
        winner_logits = logits_b if keep_cl else logits_a
        if not keep_cl:
            graph_b, logits_b = None, None
        else:
            graph_a, logits_a = None, None
        if winner_graph is None:
            raise RuntimeError("no capturable whole-forward variant")

        winner_graph.replay()
        torch.cuda.synchronize()
        if not _matches(winner_logits, eager_ref):
            raise RuntimeError("final replay validation failed")

        plan["graph"] = winner_graph
        plan["static_logits"] = winner_logits
        plan["model"] = model
        plan["pixels"] = pixels
        plan["pixels_token"] = _pixels_token(pixels)

    @torch.inference_mode()
    def _invoke(state):
        try:
            if plan["broken"]:
                return _parent(state)
            device = state.get("device")
            if device is None or device.type != "cuda" or not torch.cuda.is_available():
                plan["broken"] = True
                return _parent(state)
            pixels = state.get("pixels")
            if not _input_ok(pixels):
                return _parent(state)
            model = state.get("model")
            if (
                plan["graph"] is None
                or model is not plan["model"]
                or _pixels_token(pixels) != plan["pixels_token"]
            ):
                _drop_graph()
                _setup(state)
            plan["graph"].replay()
            return {"logits": plan["static_logits"].clone()}
        except Exception:
            _drop_graph()
            plan["broken"] = True
            return _parent(state)

    return {PROJECT_INVOKE_CALL: _invoke}
