"""Joint candidate for the ResNet-18 offline batch-32 latency task.

Replaces the 17 selected ``ResNetConvLayer`` module boundaries (stem 7x7 layer
plus both conv positions of all 8 ``ResNetBasicLayer`` blocks, the complete
ResNetConvLayer family selected by selection.json) with an equivalent that

  1. folds each layer's eval-mode BatchNorm2d exactly into its conv
     (per output channel s = gamma / sqrt(var + eps),
      W' = W * s,  b' = beta - mu * s  [+ conv_bias * s]),
     computed once per physical module in float64 and stored as float32
     (setup work, invariant model parameters, no fixture data involved);
  2. evaluates conv(+bias) as one cudnn call, with the trailing ReLU fused
     into the same kernel via ``aten.cudnn_convolution_relu`` on the 9 sites
     whose activation is ``nn.ReLU``; the 8 ``nn.Identity`` sites use
     ``conv2d`` with the folded bias (single conv+bias kernel).

This removes the separate memory-bound eval-BN kernel
(``cudnn::bn_fw_inf_1C11_kernel_NCHW``) and the separate ReLU clamp kernel at
every selected site while keeping the identical FP32 numerical policy (the
fused op runs with TF32 disabled and benchmark disabled, matching the frozen
source configuration).  Per-site fallbacks keep the parent semantics:

  * fused op unavailable/declining at runtime -> ``conv2d``(W', b') + relu;
  * unexpected signature/dtype/device/training/grad context -> parent forward.

No global functional dispatch is installed (module-boundary callables only),
no CUDA graphs are captured, and no outputs are cached: every invocation
executes the workload on its current inputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_CUDNN_CONV_RELU = torch.ops.aten.cudnn_convolution_relu
_ROUTES = ("cudnn_conv_relu", "conv2d_folded")


def _pack_bn_into_conv(conv, bn):
    """Return (weight', bias') with eval BatchNorm folded into the conv.

    Folding is done in float64 and cast to float32 so the packed tensors are
    the closest float32 representation of the exact transform
    y = ((W*x + c) - mu) * gamma/sqrt(var+eps) + beta
      = (W*s)*x + (beta - mu*s + c*s).
    Returns None when the module layout is not supported.
    """
    if (
        not isinstance(conv, torch.nn.Conv2d)
        or not isinstance(bn, torch.nn.BatchNorm2d)
        or bn.affine is not True
        or bn.track_running_stats is not True
        or bn.weight is None
        or bn.bias is None
        or bn.running_mean is None
        or bn.running_var is None
    ):
        return None
    weight = conv.weight
    if weight is None or weight.dtype is not torch.float32 or not weight.is_cuda:
        return None
    with torch.no_grad():
        eps = float(bn.eps)
        scale = bn.weight.double() / (bn.running_var.double() + eps).sqrt()
        packed_weight = (weight.double() * scale.reshape(-1, 1, 1, 1)).to(torch.float32).contiguous()
        bias64 = bn.bias.double() - bn.running_mean.double() * scale
        if conv.bias is not None:
            bias64 = bias64 + conv.bias.double() * scale
        packed_bias = bias64.to(torch.float32).contiguous()
    return packed_weight, packed_bias


def _probe_fused_op(device):
    """One-time installation check that aten.cudnn_convolution_relu services
    this device for plain FP32 NCHW convolutions.  Setup-only, tiny, and
    synchronized so asynchronous failures surface here, not in a later call."""
    try:
        with torch.no_grad():
            x = torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float32)
            w = torch.zeros(1, 1, 3, 3, device=device, dtype=torch.float32)
            b = torch.zeros(1, device=device, dtype=torch.float32)
            out = _CUDNN_CONV_RELU(x, w, b, [1, 1], [1, 1], [1, 1], 1)
            torch.cuda.synchronize(device)
            return tuple(out.shape) == (1, 1, 4, 4)
    except Exception:
        return False


def build(context):
    modules = context["modules"]
    references = context["references"]
    record_route = context.get("record_route") or (lambda *_: None)

    parameters = context.get("parameters") or {}
    requested_route = parameters.get("conv_route", "cudnn_conv_relu")
    if requested_route not in _ROUTES:
        requested_route = "cudnn_conv_relu"

    replacements = {}

    first_device = None
    for call_id, module in modules.items():
        conv = getattr(module, "convolution", None)
        bn = getattr(module, "normalization", None)
        act = getattr(module, "activation", None)
        if conv is None or bn is None or act is None:
            continue
        if not isinstance(act, (torch.nn.ReLU, torch.nn.Identity)):
            continue
        packed = _pack_bn_into_conv(conv, bn)
        if packed is None:
            continue
        if first_device is None:
            first_device = packed[0].device
        replacements[call_id] = _make_replacement(
            call_id=call_id,
            parent=references[call_id],
            conv=conv,
            bn=bn,
            relu_epilogue=isinstance(act, torch.nn.ReLU),
            packed=packed,
            requested_route=requested_route,
            record_route=record_route,
        )

    fused_available = False
    if first_device is not None and replacements:
        fused_available = _probe_fused_op(first_device)
    for replacement in replacements.values():
        replacement._opop_set_fused_available(fused_available)
    return replacements


def _make_replacement(call_id, parent, conv, bn, relu_epilogue, packed, requested_route, record_route):
    packed_weight, packed_bias = packed
    stride = tuple(int(s) for s in conv.stride)
    padding = tuple(int(p) for p in conv.padding)
    dilation = tuple(int(d) for d in conv.dilation)
    groups = int(conv.groups)
    packed_device = packed_weight.device

    # Per-site routing state, decided at installation and latched at runtime.
    state = {
        "use_fused": bool(relu_epilogue and requested_route == "cudnn_conv_relu"),
        "fused_failed": False,
        "setup_prefix": "",  # set by build() when the fused probe declines
    }

    def replacement(*args, **kwargs):
        hidden = None
        if len(args) == 1 and not kwargs:
            hidden = args[0]
        elif not args and len(kwargs) == 1 and "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
        else:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)

        if not (
            torch.is_tensor(hidden)
            and hidden.dim() == 4
            and hidden.dtype is torch.float32
            and hidden.is_cuda
            and hidden.device == packed_device
        ):
            record_route(call_id, "parent:unsupported_signature")
            return parent(hidden)
        if conv.training or bn.training:
            record_route(call_id, "parent:training_mode")
            return parent(hidden)
        if torch.is_grad_enabled() and hidden.requires_grad:
            record_route(call_id, "parent:grad_context")
            return parent(hidden)

        if state["use_fused"] and not state["fused_failed"]:
            try:
                out = _CUDNN_CONV_RELU(
                    hidden, packed_weight, packed_bias,
                    list(stride), list(padding), list(dilation), groups,
                )
                record_route(call_id, "aten.cudnn_convolution_relu")
                return out
            except Exception:
                state["fused_failed"] = True
                record_route(call_id, "fused_runtime_error->conv2d_folded+relu")

        if relu_epilogue:
            out = F.conv2d(hidden, packed_weight, packed_bias, stride, padding, dilation, groups)
            record_route(call_id, state["setup_prefix"] + "conv2d_folded+relu")
            return torch.relu(out)
        out = F.conv2d(hidden, packed_weight, packed_bias, stride, padding, dilation, groups)
        record_route(call_id, "conv2d_folded")
        return out

    def _opop_set_fused_available(available):
        if state["use_fused"] and not available:
            state["use_fused"] = False
            state["setup_prefix"] = "setup:fused_unavailable->"

    replacement._opop_set_fused_available = _opop_set_fused_available
    return replacement
