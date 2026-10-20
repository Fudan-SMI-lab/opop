from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

_BLOCK = 1024
_NUM_WARPS = 4

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
    def _bn_relu_kernel(X, A, B, Y, hw, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        c = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hw
        base = c * hw
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        a = tl.load(A + c)
        b = tl.load(B + c)
        y = tl.maximum(x * a + b, 0.0)
        tl.store(Y + base + offs, y, mask=mask)

    @triton.jit
    def _bn_add_relu_kernel(X, A, B, S, Y, hw, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        c = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < hw
        base = c * hw
        x = tl.load(X + base + offs, mask=mask, other=0.0)
        s = tl.load(S + base + offs, mask=mask, other=0.0)
        a = tl.load(A + c)
        b = tl.load(B + c)
        y = tl.maximum(x * a + b + s, 0.0)
        tl.store(Y + base + offs, y, mask=mask)


class _Fallback(Exception):
    pass


def _bn_affine(bn):
    if (
        bn is None
        or bn.training
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
    return a.detach(), b.detach()


def _eligible_tensor(x):
    return (
        torch.is_tensor(x)
        and x.is_cuda
        and x.dim() == 4
        and x.dtype == torch.float32
        and x.is_contiguous()
        and x.numel() > 0
    )


def _launch_bn_relu(x, a, b, out):
    n, c, h, w = x.shape
    hw = n * h * w
    grid = (triton.cdiv(hw, _BLOCK), c)
    _bn_relu_kernel[grid](x, a, b, out, hw, BLOCK=_BLOCK, num_warps=_NUM_WARPS)


def _launch_bn_add_relu(x, a, b, s, out):
    n, c, h, w = x.shape
    hw = n * h * w
    grid = (triton.cdiv(hw, _BLOCK), c)
    _bn_add_relu_kernel[grid](x, a, b, s, out, hw, BLOCK=_BLOCK, num_warps=_NUM_WARPS)


def _parent_only(call_id, reference, record_route):
    def forward(*args, **kwargs):
        record_route(call_id, "setup_fallback_parent")
        return reference(*args, **kwargs)

    return forward


def _build_stem(call_id, mod, reference, record_route):
    if tl is None:
        return _parent_only(call_id, reference, record_route)
    try:
        conv = mod.embedder.convolution
        bn = mod.embedder.normalization
        act = mod.embedder.activation
        pooler = mod.pooler
        num_channels = int(mod.num_channels)
    except (AttributeError, TypeError, ValueError):
        return _parent_only(call_id, reference, record_route)
    if (
        not isinstance(conv, nn.Conv2d)
        or not isinstance(bn, nn.BatchNorm2d)
        or not isinstance(act, nn.ReLU)
        or not isinstance(pooler, nn.MaxPool2d)
    ):
        return _parent_only(call_id, reference, record_route)
    packed = _bn_affine(bn)
    if packed is None:
        return _parent_only(call_id, reference, record_route)
    a, b = packed
    channels = a.numel()

    def forward(*args, **kwargs):
        x = args[0] if args else kwargs.get("pixel_values")
        try:
            if (
                not _eligible_tensor(x)
                or int(x.shape[1]) != num_channels
                or int(x.shape[1]) != channels
            ):
                raise _Fallback
            emb = conv(x)
            if not _eligible_tensor(emb):
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
        return _parent_only(call_id, reference, record_route)
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
        return _parent_only(call_id, reference, record_route)
    if (
        not isinstance(conv1, nn.Conv2d)
        or not isinstance(bn1, nn.BatchNorm2d)
        or not isinstance(act1, nn.ReLU)
        or not isinstance(conv2, nn.Conv2d)
        or not isinstance(bn2, nn.BatchNorm2d)
        or not isinstance(act, nn.ReLU)
    ):
        return _parent_only(call_id, reference, record_route)
    packed1 = _bn_affine(bn1)
    packed2 = _bn_affine(bn2)
    if packed1 is None or packed2 is None:
        return _parent_only(call_id, reference, record_route)
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
            return _parent_only(call_id, reference, record_route)
        if not isinstance(sc_conv, nn.Conv2d) or not isinstance(sc_bn, nn.BatchNorm2d):
            return _parent_only(call_id, reference, record_route)

    def forward(*args, **kwargs):
        x = args[0] if args else kwargs.get("hidden_state")
        try:
            if not _eligible_tensor(x):
                raise _Fallback
            h = conv1(x)
            if not _eligible_tensor(h) or int(h.shape[1]) != c1:
                raise _Fallback
            y1 = torch.empty_like(h)
            _launch_bn_relu(h, a1, b1, y1)
            z = conv2(y1)
            if not _eligible_tensor(z) or int(z.shape[1]) != c2:
                raise _Fallback
            s = x if identity_shortcut else shortcut(x)
            if not torch.is_tensor(s) or not s.is_contiguous() or tuple(s.shape) != tuple(z.shape):
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
