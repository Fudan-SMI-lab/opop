"""Composite candidate for the selected ResNet conv->BN(->ReLU) region.

Each selected ResNetConvLayer / ResNetShortCut site executes as one folded
convolution: eval-mode BatchNorm2d statistics are folded into the conv weight
and a derived bias, and bias(+relu where the site has an activation) runs in
the cudnn epilogue when the calibrated fused path is numerically equivalent
and not slower; otherwise F.conv2d on the folded tensors is used. Every site
keeps a guarded fallback to its parent forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ATOL = 1e-4
_RTOL = 1e-4


def _cudnn_conv_relu_op():
    try:
        op = torch.ops.aten.cudnn_convolution_relu
        _ = op.default._schema
        return op
    except Exception:
        return None


_CONV_RELU_OP = _cudnn_conv_relu_op()

_REGION_SEMANTICS = (
    "ResNetConvLayer",
    "ResNetShortCut",
)


def _fold(conv, bn):
    weight = conv.weight
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    if weight is None or gamma is None or beta is None or running_mean is None or running_var is None:
        return None
    with torch.no_grad():
        scale = (gamma / torch.sqrt(running_var + bn.eps)).to(dtype=weight.dtype)
        folded_w = (weight * scale.view(-1, 1, 1, 1)).contiguous()
        folded_b = beta - running_mean * scale
        if conv.bias is not None:
            folded_b = folded_b + conv.bias * scale
        folded_b = folded_b.contiguous()
    return folded_w, folded_b


def _close(actual, ref):
    try:
        if actual.shape != ref.shape or actual.dtype != ref.dtype or actual.device != ref.device:
            return False
        diff = (actual - ref).abs()
        tol = _ATOL + _RTOL * ref.abs()
        return bool(torch.all(diff <= tol).item())
    except Exception:
        return False


def _time_ms(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _observed_shape(spec, rank):
    for signatures in (
        (spec.get("rank_specs") or {}).get(str(rank), {}).get("signatures"),
        spec.get("signatures"),
    ):
        if not signatures:
            continue
        args = signatures[0].get("args") or []
        if args and isinstance(args[0], dict) and args[0].get("shape"):
            shape = [int(v) for v in args[0]["shape"]]
            if len(shape) == 4:
                return shape
    return None


def _calibrate(call_id, conv, bn, parent, fuse_relu, shape):
    if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
        return "parent", "setup_unsupported_normalization"
    if not getattr(bn, "track_running_stats", False):
        return "parent", "setup_fold_unavailable"
    if fuse_relu is None:
        return "parent", "setup_unsupported_activation"
    device = conv.weight.device
    dtype = conv.weight.dtype
    stride = [int(v) for v in conv.stride]
    padding = [int(v) for v in conv.padding]
    dilation = [int(v) for v in conv.dilation]
    groups = int(conv.groups)
    in_ch = int(conv.in_channels)
    if shape is None or shape[1] != in_ch:
        kernel = max(int(conv.kernel_size[0]), 1)
        shape = [2, in_ch, 8 * kernel + 8, 8 * kernel + 8]
    seed = 0x51ED2705 + (sum(ord(c) for c in call_id) % 1000003)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    try:
        with torch.no_grad():
            folded = _fold(conv, bn)
            if folded is None:
                return "parent", "setup_fold_unavailable"
            w, b = folded
            ref = parent(x)
            if not isinstance(ref, torch.Tensor):
                return "parent", "setup_parent_signature"

            def run_conv2d():
                out = F.conv2d(x, w, b, stride, padding, dilation, groups)
                if fuse_relu:
                    out = torch.relu_(out)
                return out

            candidates = {"conv2d_folded": run_conv2d}
            if fuse_relu and _CONV_RELU_OP is not None:

                def run_fused():
                    return _CONV_RELU_OP(x, w, b, stride, padding, dilation, groups)

                candidates["cudnn_fused"] = run_fused

            compliant = {}
            for name, fn in candidates.items():
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    if _close(y, ref):
                        compliant[name] = fn
                    del y
                except Exception:
                    continue
            if not compliant:
                return "parent", "setup_calibration_failed"
            parent_ms = None
            try:
                parent_ms = _time_ms(lambda: parent(x))
            except Exception:
                parent_ms = None
            times = {name: _time_ms(fn) for name, fn in compliant.items()}
            best = min(times, key=times.get)
            if "cudnn_fused" in times and times["cudnn_fused"] <= times[best] * 1.02:
                best = "cudnn_fused"
            if parent_ms is not None and parent_ms < times[best] * 0.90:
                return "parent", "setup_parent_fastest"
            return best, "calibrated:" + best
    finally:
        del x
        torch.cuda.synchronize()


def _make_replacement(call_id, module, parent, record_route, tier, setup_label, fuse_relu):
    conv = module.convolution
    bn = module.normalization
    stride = [int(v) for v in conv.stride]
    padding = [int(v) for v in conv.padding]
    dilation = [int(v) for v in conv.dilation]
    groups = int(conv.groups)
    in_ch = int(conv.in_channels)
    device = conv.weight.device
    cache = {"key": None, "w": None, "b": None}

    def _ensure_fold():
        bias = conv.bias
        key = (
            conv.weight._version,
            bn.weight._version,
            bn.bias._version,
            bn.running_mean._version,
            bn.running_var._version,
            -1 if bias is None else bias._version,
        )
        if cache["key"] != key:
            folded = _fold(conv, bn)
            cache["w"], cache["b"] = folded if folded is not None else (None, None)
            cache["key"] = key
        return cache["w"], cache["b"]

    def replacement(*args, **kwargs):
        if tier == "parent":
            record_route(call_id, "setup_route_parent:" + setup_label)
            return parent(*args, **kwargs)
        x = None
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
        else:
            for name in ("hidden_states", "input"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor):
                    x = value
                    break
        if (
            x is None
            or x.dim() != 4
            or x.dtype != torch.float32
            or x.device != device
            or int(x.shape[1]) != in_ch
            or module.training
            or conv.training
            or bn.training
            or torch.is_grad_enabled()
        ):
            record_route(call_id, "guard_unsupported_signature_parent")
            return parent(*args, **kwargs)
        w, b = _ensure_fold()
        if w is None or b is None:
            record_route(call_id, "guard_fold_unavailable_parent")
            return parent(*args, **kwargs)
        try:
            if tier == "cudnn_fused":
                out = _CONV_RELU_OP(x, w, b, stride, padding, dilation, groups)
                record_route(call_id, "cudnn_conv_bias_relu_fused")
                return out
            out = F.conv2d(x, w, b, stride, padding, dilation, groups)
            if fuse_relu:
                torch.relu_(out)
            record_route(call_id, "conv2d_bn_folded_relu" if fuse_relu else "conv2d_bn_folded")
            return out
        except Exception:
            record_route(call_id, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def build(context):
    record_route = context.get("record_route") or (lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    targets = context.get("targets") or {}
    rank = context.get("rank", 0)
    replacements = {}
    for call_id, module in modules.items():
        try:
            if module is None:
                continue
            spec = targets.get(call_id) or {}
            semantics = str(spec.get("semantics") or "")
            if not any(semantics.endswith(name) for name in _REGION_SEMANTICS):
                continue
            parent = references.get(call_id)
            if parent is None:
                continue
            conv = getattr(module, "convolution", None)
            bn = getattr(module, "normalization", None)
            act = getattr(module, "activation", None)
            if not isinstance(conv, nn.Conv2d) or bn is None:
                replacements[call_id] = _make_replacement(
                    call_id, module, parent, record_route, "parent", "setup_unsupported_module", False
                )
                continue
            if isinstance(act, nn.ReLU):
                fuse_relu = True
            elif act is None or isinstance(act, nn.Identity):
                fuse_relu = False
            else:
                fuse_relu = None
            shape = _observed_shape(spec, rank)
            try:
                tier, label = _calibrate(call_id, conv, bn, parent, fuse_relu, shape)
            except Exception:
                tier, label = "parent", "setup_exception_parent"
            record_route(call_id, "setup:" + label)
            replacements[call_id] = _make_replacement(call_id, module, parent, record_route, tier, label, fuse_relu)
        except Exception:
            parent = references.get(call_id)
            if parent is not None:
                replacements[call_id] = _make_replacement(
                    call_id, module, parent, record_route, "parent", "setup_exception_parent", False
                )
    return replacements
