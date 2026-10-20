"""Whole-bundle replacements for ResNet-18 stages 0 and 1 (memory project).

The accepted stem bundle (trial-61fdc27afc-param-cc83352f) left the project peak
at 205,071,872 B over a 76,749,312 B standing allocation: a transient of
128,322,560 B ~= 5 x 25,690,112 B, the exact size of one [32,64,56,56] fp32
activation.  The only code producing that shape is the first two encoder
stages (selection rationale, revision a72bba2ff): stages.0 keeps its [32,64,56,56]
input live for the whole stage call (ResNetStage.forward binds ``input`` for the
full loop) while ResNetConvLayer.forward materializes out-of-place conv, BN and
ReLU buffers of the same size, and stages.1.layers.0 keeps stage0's output live
as residual/shortcut input next to its [32,128,28,28] chain.

This candidate extends the accepted stem mechanism to both stages as one
coherent region (two whole-bundle replacements at the two stage boundaries of
anchor group 928c554228ed389eec5f):

  * at install time, fold each eval-mode BatchNorm affine
    (scale = gamma*rsqrt(running_var+eps), shift = beta - running_mean*scale
    [+ conv_bias*scale]) into a private copy of the preceding conv weight/bias
    (4 folded pairs in stages.0, 5 in stages.1 including the 1x1 shortcut conv
    of stages.1.layers.0); module parameters and running buffers are never
    mutated;
  * per invocation, recompute the whole stage on the current input batch-slice
    by batch-slice: folded conv -> in-place relu_ -> folded conv -> the residual
    add written straight into a preallocated stage output slice (``add_`` on the
    output slice for Identity shortcuts, ``torch.add(..., out=...)`` for conv
    shortcuts) -> in-place relu_, so the out-of-place BN/ReLU buffers and the
    full-batch conv transients disappear and the largest transient becomes
    slice-sized.

Every invocation recomputes both stages on the current input tensor.  Any
signature/state mismatch (non-fp32/CPU/non-contiguous/wrong rank/channels,
training mode), any setup problem, or any runtime error inside the fast path
falls back to the parent forward with a distinct route label; the input tensor
and module state are never mutated, so fallback is exact.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

STAGE_CALLS = {
    0: "24897547bc26374acdc8a6ea",  # model.resnet.encoder.stages.0
    1: "45bd63d6048d026c10970985",  # model.resnet.encoder.stages.1
}
DEFAULT_CHUNK = 8


def _folded_params(conv, bn):
    """Return private (weight, bias) copies with the eval-mode BN affine folded in."""
    weight = conv.weight.detach()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma * torch.rsqrt(var + float(bn.eps))
    shift = beta - mean * scale
    if conv.bias is not None:
        shift = shift + conv.bias.detach() * scale
    view = (1,) * (weight.dim() - 1)
    folded_weight = (weight * scale.view(-1, *view)).contiguous()
    return folded_weight, shift.contiguous()


def _fold_conv_bn(conv, bn):
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError("stage conv member is not Conv2d")
    if not isinstance(bn, torch.nn.BatchNorm2d):
        raise TypeError("stage normalization member is not BatchNorm2d")
    if conv.padding_mode != "zeros":
        raise TypeError("unsupported conv padding mode")
    if not (getattr(bn, "affine", False) and getattr(bn, "track_running_stats", False)):
        raise TypeError("unsupported BatchNorm2d configuration")
    if bn.running_mean is None or bn.running_var is None:
        raise TypeError("missing BN running stats")
    if bn.num_features != conv.out_channels:
        raise TypeError("BN feature count mismatch")
    with torch.inference_mode(False), torch.no_grad():
        folded_weight, folded_bias = _folded_params(conv, bn)
    return {
        "weight": folded_weight,
        "bias": folded_bias,
        "kwargs": {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        },
    }


def _activation_kind(act):
    if isinstance(act, torch.nn.Identity):
        return "identity"
    if isinstance(act, torch.nn.ReLU):
        return "relu"
    if act is F.relu or getattr(act, "__name__", "") == "relu":
        return "relu"
    raise TypeError("unsupported activation")


def _build_stage_plan(stage):
    """Validate a ResNetStage of BasicLayers and fold every conv+BN pair."""
    layers = list(stage.layers)
    if not layers:
        raise TypeError("stage has no layers")
    plan = {"layers": [], "bns": []}
    first_conv = layers[0].layer[0].convolution
    plan["in_channels"] = int(first_conv.in_channels)
    for basic in layers:
        conv_layers = list(basic.layer)
        if len(conv_layers) != 2:
            raise TypeError("basic layer does not have exactly two conv layers")
        entry = {"convs": [], "shortcut": None}
        for conv_layer in conv_layers:
            folded = _fold_conv_bn(conv_layer.convolution, conv_layer.normalization)
            folded["act"] = _activation_kind(conv_layer.activation)
            entry["convs"].append(folded)
            plan["bns"].append(conv_layer.normalization)
        if _activation_kind(basic.activation) != "relu":
            raise TypeError("basic layer final activation is not ReLU")
        shortcut = basic.shortcut
        if isinstance(shortcut, torch.nn.Identity):
            entry["shortcut"] = None
        else:
            entry["shortcut"] = _fold_conv_bn(shortcut.convolution, shortcut.normalization)
            plan["bns"].append(shortcut.normalization)
        plan["layers"].append(entry)
    return plan


def _run_stage(plan, x, chunk):
    """Recompute the whole stage slice-wise with folded convs (fast path body)."""
    batch = x.shape[0]
    out = None
    start = 0
    while start < batch:
        stop = min(start + chunk, batch)
        h = x[start:stop]
        out_s = None
        for entry in plan["layers"]:
            conv0, conv1 = entry["convs"]
            t = F.conv2d(h, conv0["weight"], conv0["bias"], **conv0["kwargs"])
            if conv0["act"] == "relu":
                t.relu_()
            t2 = F.conv2d(t, conv1["weight"], conv1["bias"], **conv1["kwargs"])
            if conv1["act"] == "relu":
                t2.relu_()
            shortcut = entry["shortcut"]
            res = F.conv2d(h, shortcut["weight"], shortcut["bias"], **shortcut["kwargs"]) if shortcut else h
            if out is None:
                out = torch.empty((batch,) + tuple(t2.shape)[1:], device=t2.device, dtype=t2.dtype)
            out_s = out[start:stop]
            if res is out_s:
                out_s.add_(t2)
            else:
                torch.add(t2, res, out=out_s)
            out_s.relu_()
            h = out_s
        start = stop
    return out


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parameters = context.get("parameters") or {}

    chunk = parameters.get("stage_chunk", DEFAULT_CHUNK)
    try:
        chunk = max(1, int(chunk))
    except (TypeError, ValueError):
        chunk = DEFAULT_CHUNK

    replacements = {}
    for idx, call_id in STAGE_CALLS.items():
        module = modules.get(call_id)
        if module is None:
            continue  # no local replacement declared for this site on this rank
        parent = references.get(call_id)
        if parent is None:
            parent = module.forward

        try:
            plan = _build_stage_plan(module)
        except Exception:
            plan = None  # reported per call as a setup-failure route, never silent

        def replacement(*args, _call_id=call_id, _idx=idx, _module=module, _plan=plan, _parent=parent, **kwargs):
            if _plan is None:
                record_route(_call_id, f"stage{_idx}:setup-failure:parent")
                return _parent(*args, **kwargs)
            x = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            if (
                not torch.is_tensor(x)
                or x.dim() != 4
                or not x.is_cuda
                or x.dtype != torch.float32
                or x.shape[1] != _plan["in_channels"]
                or not x.is_contiguous()
                or x.requires_grad
                or x.shape[0] == 0
                or _module.training
                or any(bn.training for bn in _plan["bns"])
            ):
                record_route(_call_id, f"stage{_idx}:unsupported-signature:parent")
                return _parent(*args, **kwargs)
            try:
                out = _run_stage(_plan, x, chunk)
                record_route(
                    _call_id,
                    f"stage{_idx}:folded-single-slice" if x.shape[0] <= chunk else f"stage{_idx}:folded-chunked",
                )
                return out
            except Exception:
                record_route(_call_id, f"stage{_idx}:runtime-failure:parent")
                return _parent(*args, **kwargs)

        replacements[call_id] = replacement
    return replacements
