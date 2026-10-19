"""Joint candidate for the frozen ResNet-18 FP32 batch=1 online inference workload.

Replaces each of the 20 selected conv->BatchNorm2d epilogue sites (17
ResNetConvLayer modules + 3 ResNetShortCut modules) with one fused convolution
that absorbs the eval-mode BatchNorm affine:

    s      = gamma / sqrt(running_var + eps)
    W_fold = W * s                  (per output channel)
    b_fold = beta - s * running_mean (+ s * conv_bias when a conv bias exists)
    y      = act(conv2d(x, W_fold, b_fold))

which is algebraically identical to act(bn(conv(x))) for training=False,
track_running_stats=True BatchNorm in exact arithmetic. Folding arithmetic runs
in float64 and the results are stored as new FP32 tensors; the model's own
parameters, buffers and structure are never mutated. Calls whose input does not
match the observed per-site signature (shape/stride/dtype/device), or that carry
extra args/kwargs, fall back to the module's original bound forward captured at
build time, so unobserved inputs keep exact parent semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_BatchNorm = torch.nn.modules.batchnorm._BatchNorm


def _expected_dtype(name):
    if not isinstance(name, str) or not name.startswith("torch."):
        return None
    return getattr(torch, name[len("torch."):], None)


def _signature_matches(tensor, spec):
    if not isinstance(tensor, torch.Tensor):
        return False
    shape = spec.get("shape")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        return False
    stride = spec.get("stride")
    if stride is not None and tensor.stride() != tuple(stride):
        return False
    dtype = _expected_dtype(spec.get("dtype"))
    if dtype is not None and tensor.dtype != dtype:
        return False
    device = spec.get("device")
    if device == "cuda" and not tensor.is_cuda:
        return False
    return True


def _fold_site(conv, bn):
    if conv.padding_mode != "zeros":
        return None
    if bn.training or not bn.track_running_stats or not bn.affine:
        return None
    if bn.running_mean is None or bn.running_var is None:
        return None
    if conv.weight is None or bn.weight is None or bn.bias is None:
        return None
    with torch.no_grad():
        weight = conv.weight.detach()
        work = dict(device=weight.device, dtype=torch.float64)
        gamma = bn.weight.detach().to(**work)
        beta = bn.bias.detach().to(**work)
        mean = bn.running_mean.detach().to(**work)
        var = bn.running_var.detach().to(**work)
        scale = gamma / torch.sqrt(var + float(bn.eps))
        shift = beta - scale * mean
        folded_weight = (weight.to(torch.float64) * scale.reshape(-1, 1, 1, 1)).to(weight.dtype).contiguous()
        if conv.bias is not None:
            folded_bias = (shift + scale * conv.bias.detach().to(torch.float64).reshape(-1)).to(weight.dtype).contiguous()
        else:
            folded_bias = shift.to(weight.dtype).contiguous()
    return folded_weight, folded_bias


def _make_replacement(weight, bias, stride, padding, dilation, groups, use_relu, spec, reference):
    def fused_forward(*args, **kwargs):
        if len(args) == 1 and not kwargs and _signature_matches(args[0], spec):
            out = F.conv2d(args[0], weight, bias, stride, padding, dilation, groups)
            if use_relu:
                out = torch.relu(out)
            return out
        return reference(*args, **kwargs)

    return fused_forward


def build(context):
    replacements = {}
    if not isinstance(context, dict):
        return replacements
    targets = context.get("targets") or {}
    modules = context.get("modules") or {}
    references = context.get("references") or {}

    for call_id, spec in targets.items():
        if not isinstance(spec, dict) or spec.get("kind") != "module":
            continue
        semantics = spec.get("semantics") or ""
        if not (semantics.endswith("ResNetConvLayer") or semantics.endswith("ResNetShortCut")):
            continue
        module = modules.get(call_id)
        if not isinstance(module, torch.nn.Module):
            continue
        children = list(module.children())
        if len(children) not in (2, 3):
            continue
        conv, bn = children[0], children[1]
        if not isinstance(conv, torch.nn.Conv2d) or not isinstance(bn, _BatchNorm):
            continue
        use_relu = False
        if len(children) == 3:
            if isinstance(children[2], torch.nn.ReLU):
                use_relu = True
            elif not isinstance(children[2], torch.nn.Identity):
                continue
        signatures = spec.get("signatures") or []
        if not signatures:
            continue
        args_spec = (signatures[0].get("args") or [None])[0]
        if not isinstance(args_spec, dict):
            continue
        folded = _fold_site(conv, bn)
        if folded is None:
            continue
        weight, bias = folded
        reference = module.forward
        if not callable(reference):
            reference = references.get(call_id)
        replacements[call_id] = _make_replacement(
            weight,
            bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            use_relu,
            args_spec,
            reference,
        )
    return replacements
