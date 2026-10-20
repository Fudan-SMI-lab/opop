"""Fused pointwise chains for the frozen ResNet-18 offline batch=32 region.

Replaces the forward of ResNetEmbeddings (stem) and the 8 ResNetBasicLayer
modules with fused eval-BN+ReLU / eval-BN+add+ReLU Triton kernels, leaving
every convolution, the stem max-pool and the shortcut modules on their parent
dispatch (unchanged cudnn kernels and outputs).

Repair of trial-77617c1424, whose quality check failed with
max_abs_error 40.84 on 32000/32000 logits:

1. Kernel indexing bug: the previous kernels launched grid axis 1 = channel
   index c and computed the plane base as ``c * (N*H*W)``, i.e. they indexed
   the tensors channel-major [C, N*H*W]. A contiguous NCHW tensor is laid out
   batch-major: element (n, ch, h, w) lives at (n*C + ch)*H*W + h*W + w. The
   mismatch scrambled every channel/batch pair (32 rows x 1000 classes all
   wrong). Fix: grid axis 1 enumerates N*C output planes; plane p covers the
   contiguous span [p*H*W, (p+1)*H*W) and its channel index is p % C, which
   addresses the packed A/B affine vectors.

2. Stem guard bug: the previous stem guard required
   ``x.shape[1] == num_channels and x.shape[1] == channels`` where
   num_channels=3 (input) and channels=64 (BN size) can never hold at once,
   so the stem silently fell back to the parent (never fused, evidence:
   route 'parent_fallback_guard' x2 in the trial's coverage). Fix: the input
   is checked against num_channels and the conv output against the BN size.

Numerics are the same eval-BN affine the native inference kernel evaluates:
a = weight * rsqrt(running_var + eps), b = bias - running_mean * a, packed
once at build from live parameters (no parameter mutation), then
y = max(fma(x, a, b), 0) and y = max(fma(z, a2, b2) + s, 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - triton ships with the pinned worker
    triton = None
    tl = None


STEM_CALL_ID = "683c822136ed7958f3e9ba44"
BLOCK_CALL_IDS = (
    "1b54ad939831022e60edccb3",
    "e75f33574334e8fdb1ec1732",
    "a5217ba2e7b6aec29e7666ed",
    "f3100f43551c26ea1408b2c6",
    "769c13d3cea59401691724f2",
    "33a0ef109fa92d5f85fef188",
    "7157dbaaf35beb11e060b177",
    "4853d54385e5d0d5898b92f0",
)


if tl is not None:

    @triton.jit
    def _bn_relu_kernel(X, A, B, Y, hw, C, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.program_id(1)
        c = p % C
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hw
        base = p * hw
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        a = tl.load(A + c)
        b = tl.load(B + c)
        y = tl.maximum(x * a + b, 0.0)
        tl.store(Y + base + offs, y, mask=mask)

    @triton.jit
    def _bn_add_relu_kernel(X, A, B, S, Y, hw, C, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = tl.program_id(1)
        c = p % C
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hw
        base = p * hw
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        s = tl.load(S + base + offs, mask=mask, other=0.0)
        a = tl.load(A + c)
        b = tl.load(B + c)
        y = tl.maximum(x * a + b + s, 0.0)
        tl.store(Y + base + offs, y, mask=mask)


class _Fallback(Exception):
    """Guard trip: re-dispatch this call to the parent forward."""


def _plane_config(hw):
    """Block size / warps for a launch over planes of hw contiguous elements."""
    if hw <= 64:
        return 64, 2
    if hw <= 256:
        return 256, 4
    if hw <= 512:
        return 512, 4
    return 1024, 4


def _launch_bn_relu(x, a, b, out):
    n, c, h, w = x.shape
    hw = int(h) * int(w)
    planes = int(n) * int(c)
    block, warps = _plane_config(hw)
    grid = (triton.cdiv(hw, block), planes)
    _bn_relu_kernel[grid](x, a, b, out, hw, int(c), BLOCK=block, num_warps=warps)


def _launch_bn_add_relu(x, a, b, s, out):
    n, c, h, w = x.shape
    hw = int(h) * int(w)
    planes = int(n) * int(c)
    block, warps = _plane_config(hw)
    grid = (triton.cdiv(hw, block), planes)
    _bn_add_relu_kernel[grid](x, a, b, s, out, hw, int(c), BLOCK=block, num_warps=warps)


def _bn_affine(bn):
    """Pack eval-BN into y = x * a + b from live parameters (read-only)."""
    if (
        bn is None
        or bn.training
        or not getattr(bn, "track_running_stats", False)
        or bn.weight is None
        or bn.bias is None
        or bn.running_mean is None
        or bn.running_var is None
    ):
        return None
    with torch.no_grad():
        a = (bn.weight * torch.rsqrt(bn.running_var + bn.eps)).contiguous()
        b = (bn.bias - bn.running_mean * a).contiguous()
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        return None
    if a.numel() == 0 or b.numel() != a.numel():
        return None
    if a.device != bn.weight.device or b.device != bn.weight.device:
        return None
    return a.detach(), b.detach()


def _eligible_tensor(x):
    return (
        torch.is_tensor(x)
        and x.is_cuda
        and x.dim() == 4
        and x.dtype == torch.float32
        and x.is_contiguous()
        and 0 < x.numel() < 2**31
    )


def _single_input(args, kwargs, key):
    """The observed signatures are exactly one 4D tensor, positional or kw."""
    if len(args) == 1 and not kwargs:
        return args[0]
    if not args and len(kwargs) == 1 and key in kwargs:
        return kwargs[key]
    return None


def _parent_only(call_id, reference, record_route, label):
    def forward(*args, **kwargs):
        record_route(call_id, label)
        return reference(*args, **kwargs)

    return forward


def _build_stem(call_id, mod, reference, record_route):
    if tl is None:
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    try:
        conv = mod.embedder.convolution
        bn = mod.embedder.normalization
        act = mod.embedder.activation
        pooler = mod.pooler
        num_channels = int(mod.num_channels)
    except (AttributeError, TypeError, ValueError):
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    if (
        not isinstance(conv, nn.Conv2d)
        or not isinstance(bn, nn.BatchNorm2d)
        or not isinstance(act, nn.ReLU)
        or not isinstance(pooler, nn.MaxPool2d)
    ):
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    packed = _bn_affine(bn)
    if packed is None:
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    a, b = packed
    channels = a.numel()
    if int(conv.out_channels) != channels:
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")

    def forward(*args, **kwargs):
        x = _single_input(args, kwargs, "pixel_values")
        try:
            if (
                x is None
                or not _eligible_tensor(x)
                or int(x.shape[1]) != num_channels
            ):
                raise _Fallback
            emb = conv(x)
            if (
                not _eligible_tensor(emb)
                or int(emb.shape[1]) != channels
                or int(emb.shape[0]) != int(x.shape[0])
            ):
                raise _Fallback
            out = torch.empty_like(emb)
            _launch_bn_relu(emb, a, b, out)
            pooled = pooler(out)
            record_route(call_id, "triton_fused_bn_relu")
            return pooled
        except _Fallback:
            record_route(call_id, "parent_fallback_guard")
            return reference(*args, **kwargs)
        except Exception:
            record_route(call_id, "parent_fallback_error")
            return reference(*args, **kwargs)

    return forward


def _build_block(call_id, mod, reference, record_route):
    if tl is None:
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    try:
        layer0 = mod.layer[0]
        layer1 = mod.layer[1]
        conv1 = layer0.convolution
        bn1 = layer0.normalization
        act1 = layer0.activation
        conv2 = layer1.convolution
        bn2 = layer1.normalization
        shortcut = mod.shortcut
        act = mod.activation
    except (AttributeError, IndexError, TypeError):
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    if (
        not isinstance(conv1, nn.Conv2d)
        or not isinstance(bn1, nn.BatchNorm2d)
        or not isinstance(act1, nn.ReLU)
        or not isinstance(conv2, nn.Conv2d)
        or not isinstance(bn2, nn.BatchNorm2d)
        or not isinstance(layer1.activation, nn.Identity)
        or not isinstance(act, nn.ReLU)
    ):
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    packed1 = _bn_affine(bn1)
    packed2 = _bn_affine(bn2)
    if packed1 is None or packed2 is None:
        return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
    a1, b1 = packed1
    a2, b2 = packed2
    c1 = a1.numel()
    c2 = a2.numel()
    identity_shortcut = isinstance(shortcut, nn.Identity)
    if not identity_shortcut:
        try:
            sc_conv = shortcut.convolution
            sc_bn = shortcut.normalization
        except AttributeError:
            return _parent_only(call_id, reference, record_route, "setup_fallback_parent")
        if not isinstance(sc_conv, nn.Conv2d) or not isinstance(sc_bn, nn.BatchNorm2d):
            return _parent_only(call_id, reference, record_route, "setup_fallback_parent")

    def forward(*args, **kwargs):
        x = _single_input(args, kwargs, "hidden_state")
        try:
            if x is None or not _eligible_tensor(x):
                raise _Fallback
            h = conv1(x)
            if (
                not _eligible_tensor(h)
                or int(h.shape[1]) != c1
                or int(h.shape[0]) != int(x.shape[0])
            ):
                raise _Fallback
            y1 = torch.empty_like(h)
            _launch_bn_relu(h, a1, b1, y1)
            z = conv2(y1)
            if (
                not _eligible_tensor(z)
                or int(z.shape[1]) != c2
                or int(z.shape[0]) != int(x.shape[0])
            ):
                raise _Fallback
            s = x if identity_shortcut else shortcut(x)
            if (
                not torch.is_tensor(s)
                or s.dtype != torch.float32
                or s.device != z.device
                or not s.is_contiguous()
                or tuple(s.shape) != tuple(z.shape)
            ):
                raise _Fallback
            y2 = torch.empty_like(z)
            _launch_bn_add_relu(z, a2, b2, s, y2)
            record_route(call_id, "triton_fused_bn_add_relu")
            return y2
        except _Fallback:
            record_route(call_id, "parent_fallback_guard")
            return reference(*args, **kwargs)
        except Exception:
            record_route(call_id, "parent_fallback_error")
            return reference(*args, **kwargs)

    return forward


def build(context):
    modules = context.get("modules", {})
    references = context.get("references", {})
    record_route = context.get("record_route", lambda *_: None)
    replacements = {}
    stem = modules.get(STEM_CALL_ID)
    stem_reference = references.get(STEM_CALL_ID)
    if stem is not None and stem_reference is not None:
        replacements[STEM_CALL_ID] = _build_stem(
            STEM_CALL_ID, stem, stem_reference, record_route
        )
    for call_id in BLOCK_CALL_IDS:
        module = modules.get(call_id)
        reference = references.get(call_id)
        if module is None or reference is None:
            continue
        replacements[call_id] = _build_block(call_id, module, reference, record_route)
    return replacements
