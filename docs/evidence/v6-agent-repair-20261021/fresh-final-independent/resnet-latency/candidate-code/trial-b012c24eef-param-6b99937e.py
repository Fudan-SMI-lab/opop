"""Joint fused-epilogue replacements for the ResNet-18 offline latency project.

This bundle explicitly supersedes the accepted BN-fold bundle
``trial-d2a4cb48d3-param-21cc696e`` at every site it declared (17
``ResNetConvLayer`` + 3 ``ResNetShortCut`` module calls) and additionally takes
over the 8 ``ResNetBasicLayer`` module boundaries, transferring the required-call
coverage of the folded conv sites to this bundle's replacements.

Repair target (previous-feedback, revision 353b783e): the accepted fold removed
the 20 ``aten::batch_norm`` kernels but the eager cudnn path pays the folded
conv bias as a separate non-vectorized broadcast elementwise kernel (20 kernels,
355.9 us, 206.6 us alone on the [32,64,112,112] stem output), plus 17 separate
``aten::relu`` kernels (350.9 us) and 8 residual ``aten::add_`` kernels
(77.3 us): ~784 us of pointwise epilogue traffic against ~2974 us of conv
kernels on a GPU-bound invocation (kernel active union 3989.5 us of a 4071.0 us
span, host self-time ~0). The paired fold win was only ~46 us of the promised
~428 us because the new bias-add kernels replaced the removed BN kernels.

Design (same FP32 math, fused epilogues):

* 9 ReLU-terminated conv sites (stem + the 8 first convs of each basic layer)
  run the BN-folded conv as one ``aten::cudnn_convolution_relu`` execution
  (registered CUDA kernel in the worker torch 2.13.0+cu129), i.e. conv + folded
  bias + ReLU in a single cudnn kernel; the separate broadcast bias-add kernel
  and the separate ReLU kernel disappear.
* The 8 ``activation=None`` second convs and the 3 ``ResNetShortCut`` convs emit
  their BN-folded conv with ``bias=None`` (single conv kernel, no elementwise
  bias pass) when invoked inside the owning basic layer's fused window, and
  carry the folded bias into the owning ``ResNetBasicLayer`` replacement.
* Each ``ResNetBasicLayer`` replacement calls its real child module boundaries
  (``self.layer`` Sequential and ``self.shortcut``) so every declared site still
  executes on every invocation, then computes the residual epilogue
  ReLU(hidden + residual + b_conv2 + b_shortcut) as ONE vectorized Triton
  kernel, replacing conv2's bias-add, the shortcut's bias-add, the residual
  ``add_`` and the post-add ReLU.

Coordination is fail-safe: the basic-layer replacement arms a per-physical-
module window flag; a conv2/shortcut replacement returns a bias-free conv
output only when the window is armed AND its own guards pass, reporting back
through the same shared marker. Every combination of fused/fallback child paths
leaves the epilogue bias exactly the sum of the biases not already applied, and
any guard rejection anywhere falls back to the unmodified parent callable with
unchanged semantics. Guards reject training mode, enabled autograd, non-FP32,
non-CUDA or unexpected signatures at every site.

Setup work at build: exact eval-mode BN folding (scale = gamma/sqrt(var+eps),
W' = W*scale, b' = beta - scale*mean) of all 20 conv/BN pairs into contiguous
FP32 copies (~45 MB, one-time, outside timed execution), per-channel bias sums
for the fused epilogue, tiny synthetic-device probes of the cudnn fused op and
the Triton epilogue kernel (no fixture data), and lazy Triton JIT compilation
during the first quality invocations. Original parameters and BN buffers are
never modified.

Route labels reported through ``context['record_route']``:
  "fused_conv_bias_relu"          - cudnn conv+bias+ReLU single-kernel path
  "folded_conv_relu_two_op"       - conv+bias then activation (fused op unavailable)
  "coord_conv2_biasfree"          - second conv emitted without bias for the epilogue
  "folded_conv2_full"             - second conv emitted with folded bias
  "coord_shortcut_biasfree"       - shortcut conv emitted without bias
  "folded_shortcut_full"          - shortcut conv emitted with folded bias
  "basic_fused_epilogue_triton"   - basic layer with fused Triton residual epilogue
  "basic_fused_epilogue_eager"    - basic layer with eager residual epilogue
  "folded_conv_generic"           - folded conv with non-ReLU/Identity activation
  "fallback:signature"            - guard rejection, parent callable used
  "fallback:fold_setup_failed"    - fold prerequisites missing, parent used
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SITE_CALL_IDS = (
    "04de0e7be76279cb340fc51e",
    "07e19be32b79aeb05aec13d6",
    "1b160dbf6743acd183be0d7d",
    "1b163dbd8c0a67a214a9f004",
    "2cf4d2b82dc58e11738bb3b9",
    "31651bd556566af82ff06f6a",
    "3a8cb56e234373d6905df319",
    "3db154d4b6654cb04cba4425",
    "41bce2b3604affc0c6f885ae",
    "4e1cb6f216767d70f53f37cb",
    "5317ed169e344be620e52d7e",
    "66d2a06f39c8b44961a09e1b",
    "6eed86b680edffea52f8c0c7",
    "74e9871b09ae9c4a1f947695",
    "7db3c2c1041bf3fe2951687e",
    "86d53f173ec2db711c715400",
    "9000547cd2a8bfb3ff3c94e8",
    "9337819e1d7b6a87031cc9fa",
    "9c61a0898adb09fcada55871",
    "a1a1ba5d96832a39f4dab342",
    "ab5f018984612357036fff22",
    "ad0730b4f031b770865aca4c",
    "bf4bcffa4cae24a027925ba8",
    "cecf17ef310a4487f93c1329",
    "d97d55bb22b7a5b14da54326",
    "db8a4685e26c9341b26c8e7a",
    "f9ad704797a591fd3fd254a0",
    "fad32b28bd6e0def6ce7593e",
)

_CUDNN_CONV_RELU = torch.ops.aten.cudnn_convolution_relu

_TRITON_STATE = {"kernel": None, "tried": False, "failed": False}


class _Coord:
    __slots__ = ("want", "conv2_free", "shortcut_free")

    def __init__(self):
        self.want = False
        self.conv2_free = False
        self.shortcut_free = False


def _fold_conv_bn(conv, bn):
    if conv is None or bn is None:
        return None
    if getattr(conv, "training", True) or getattr(bn, "training", True):
        return None
    if not getattr(bn, "affine", False) or not getattr(bn, "track_running_stats", False):
        return None
    if bn.weight is None or bn.bias is None:
        return None
    if bn.running_mean is None or bn.running_var is None:
        return None
    if conv.weight is None or conv.groups != 1 or conv.padding_mode != "zeros":
        return None
    with torch.no_grad():
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        mean = bn.running_mean.detach()
        var = bn.running_var.detach()
        scale = gamma / torch.sqrt(var + bn.eps)
        weight = conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
        base = conv.bias.detach() if conv.bias is not None else torch.zeros_like(beta)
        bias = beta + scale * (base - mean)
        return weight.contiguous(), bias.contiguous()


def _is_conv_layer(mod):
    return (
        mod is not None
        and hasattr(mod, "convolution")
        and hasattr(mod, "normalization")
        and hasattr(mod, "activation")
    )


def _is_shortcut_conv(mod):
    return (
        mod is not None
        and hasattr(mod, "convolution")
        and hasattr(mod, "normalization")
        and not hasattr(mod, "activation")
    )


def _is_basic_layer(mod):
    if mod is None or not hasattr(mod, "layer") or not hasattr(mod, "shortcut"):
        return False
    seq = getattr(mod, "layer", None)
    try:
        return _is_conv_layer(seq[0]) and _is_conv_layer(seq[1])
    except Exception:
        return False


def _tensor_ok(t, device, in_channels):
    return (
        isinstance(t, torch.Tensor)
        and t.dim() == 4
        and t.dtype is torch.float32
        and t.device == device
        and t.shape[1] == in_channels
    )


def _triton_kernel():
    state = _TRITON_STATE
    if state["tried"] or state["failed"]:
        return state["kernel"]
    state["tried"] = True
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _add_bias_relu_kernel(x_ptr, r_ptr, b_ptr, y_ptr, n_elements, hw, channels, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            r = tl.load(r_ptr + offs, mask=mask, other=0.0)
            ch = (offs // hw) % channels
            b = tl.load(b_ptr + ch, mask=mask, other=0.0)
            tl.store(y_ptr + offs, tl.maximum(x + r + b, 0.0), mask=mask)

        state["kernel"] = _add_bias_relu_kernel
    except Exception:
        state["kernel"] = None
    return state["kernel"]


def _launch_add_bias_relu(x, residual, bias, kernel):
    out = torch.empty_like(x)
    n = x.numel()
    hw = int(x.shape[2]) * int(x.shape[3])
    channels = int(x.shape[1])
    grid = ((n + 1023) // 1024,)
    kernel[grid](x, residual, bias, out, n, hw, channels, BLOCK=1024)
    return out


def _probe_triton(device):
    kernel = _triton_kernel()
    if kernel is None or device is None or device.type != "cuda":
        return None
    try:
        x = (torch.arange(64, dtype=torch.float32, device=device) / 64.0).reshape(1, 8, 2, 4)
        residual = torch.full_like(x, 0.5)
        bias = torch.arange(8, dtype=torch.float32, device=device) / 8.0 - 0.5
        out = _launch_add_bias_relu(x, residual, bias, kernel)
        torch.cuda.synchronize()
        expected = torch.relu(x + residual + bias.view(1, -1, 1, 1))
        if out.shape == expected.shape and torch.allclose(out, expected, atol=2e-6, rtol=0.0):
            return kernel
    except Exception:
        pass
    _TRITON_STATE["failed"] = True
    _TRITON_STATE["kernel"] = None
    return None


def _probe_cudnn_conv_relu(device):
    if device is None or device.type != "cuda" or not torch.backends.cudnn.enabled:
        return False
    try:
        x = (torch.arange(96, dtype=torch.float32, device=device) / 96.0).reshape(1, 3, 4, 8)
        w = torch.zeros(4, 3, 3, 3, device=device)
        with torch.no_grad():
            w[:, :, 1, 1] = 1.0
        b = torch.zeros(4, device=device)
        y = _CUDNN_CONV_RELU(x, w, b, [1, 1], [1, 1], [1, 1], 1)
        torch.cuda.synchronize()
        expected = torch.relu(x.sum(dim=1)).unsqueeze(1)
        return bool(y.shape == (1, 4, 4, 8) and torch.allclose(y, expected, atol=1e-5, rtol=1e-5))
    except Exception:
        return False


def build(context):
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route", lambda *_: None)

    coord_for_module = {}
    fold_cache = {}
    _RUNTIME_PROBES["cudnn_conv_relu"] = None
    _RUNTIME_PROBES["triton_kernel"] = None

    def fold_of(conv, bn):
        key = id(conv)
        if key not in fold_cache:
            try:
                pair = _fold_conv_bn(conv, bn)
            except Exception:
                pair = None
            fold_cache[key] = pair
        return fold_cache[key]

    for call_id in _SITE_CALL_IDS:
        mod = modules.get(call_id)
        if mod is None or not _is_basic_layer(mod):
            continue
        conv2_mod = mod.layer[1]
        coord_for_module[id(conv2_mod)] = _Coord()
        if _is_shortcut_conv(mod.shortcut):
            coord_for_module[id(mod.shortcut)] = coord_for_module[id(conv2_mod)]

    replacements = {}
    for call_id in _SITE_CALL_IDS:
        module = modules.get(call_id)
        parent = references.get(call_id)
        if module is None or parent is None:
            continue
        if _is_basic_layer(module):
            replacements[call_id] = _build_basic_layer(
                call_id, module, parent, record_route, fold_of, coord_for_module
            )
        elif _is_shortcut_conv(module):
            replacements[call_id] = _build_shortcut_site(
                call_id, module, parent, record_route, fold_of, coord_for_module.get(id(module))
            )
        elif _is_conv_layer(module):
            replacements[call_id] = _build_conv_site(
                call_id, module, parent, record_route, fold_of, coord_for_module.get(id(module))
            )

    devices = [pair[0].device for pair in fold_cache.values() if pair is not None]
    device = devices[0] if devices else None
    _RUNTIME_PROBES["cudnn_conv_relu"] = _probe_cudnn_conv_relu(device)
    _RUNTIME_PROBES["triton_kernel"] = _probe_triton(device)
    return replacements


_RUNTIME_PROBES = {"cudnn_conv_relu": None, "triton_kernel": None}


def _build_conv_site(call_id, module, parent, record_route, fold_of, coord):
    conv = module.convolution
    bn = module.normalization
    activation = module.activation
    pair = fold_of(conv, bn)

    def setup_failed(*args, _parent=parent, _call_id=call_id, _record=record_route, **kwargs):
        _record(_call_id, "fallback:fold_setup_failed")
        return _parent(*args, **kwargs)

    if pair is None:
        return setup_failed
    weight, bias = pair
    device = weight.device
    in_channels = int(conv.in_channels)
    is_relu = isinstance(activation, torch.nn.ReLU)
    is_identity = isinstance(activation, torch.nn.Identity)

    if is_relu and coord is None:
        stride = [int(s) for s in conv.stride]
        padding = [int(p) for p in conv.padding]
        dilation = [int(d) for d in conv.dilation]
        groups = int(conv.groups)
        fused_cell = [None]

        def replacement(hidden_states, *args, _parent=parent, _call_id=call_id, _record=record_route,
                        _weight=weight, _bias=bias, _activation=activation, _conv=conv, _bn=bn,
                        _stride=stride, _padding=padding, _dilation=dilation, _groups=groups,
                        _device=device, _in_channels=in_channels, _fused=fused_cell, **kwargs):
            if (args or kwargs or torch.is_grad_enabled() or _conv.training or _bn.training
                    or not _tensor_ok(hidden_states, _device, _in_channels)
                    or not hidden_states.is_contiguous()):
                _record(_call_id, "fallback:signature")
                return _parent(hidden_states, *args, **kwargs)
            if _fused[0] is None:
                _fused[0] = _RUNTIME_PROBES["cudnn_conv_relu"]
            if _fused[0]:
                try:
                    out = _CUDNN_CONV_RELU(hidden_states, _weight, _bias, _stride, _padding, _dilation, _groups)
                    _record(_call_id, "fused_conv_bias_relu")
                    return out
                except Exception:
                    _fused[0] = False
            out = F.conv2d(hidden_states, _weight, _bias, stride=_stride, padding=_padding,
                           dilation=_dilation, groups=_groups)
            out = _activation(out)
            _record(_call_id, "folded_conv_relu_two_op")
            return out

        return replacement

    if is_identity:
        def replacement(hidden_states, *args, _parent=parent, _call_id=call_id, _record=record_route,
                        _weight=weight, _bias=bias, _conv=conv, _bn=bn, _coord=coord,
                        _device=device, _in_channels=in_channels, **kwargs):
            if (args or kwargs or torch.is_grad_enabled() or _conv.training or _bn.training
                    or not _tensor_ok(hidden_states, _device, _in_channels)):
                _record(_call_id, "fallback:signature")
                return _parent(hidden_states, *args, **kwargs)
            if _coord is not None and _coord.want:
                out = F.conv2d(hidden_states, _weight, None, stride=_conv.stride, padding=_conv.padding,
                               dilation=_conv.dilation, groups=_conv.groups)
                _coord.conv2_free = True
                _record(_call_id, "coord_conv2_biasfree")
                return out
            out = F.conv2d(hidden_states, _weight, _bias, stride=_conv.stride, padding=_conv.padding,
                           dilation=_conv.dilation, groups=_conv.groups)
            _record(_call_id, "folded_conv2_full")
            return out

        return replacement

    def replacement(hidden_states, *args, _parent=parent, _call_id=call_id, _record=record_route,
                    _weight=weight, _bias=bias, _activation=activation, _conv=conv, _bn=bn,
                    _device=device, _in_channels=in_channels, **kwargs):
        if (args or kwargs or torch.is_grad_enabled() or _conv.training or _bn.training
                or not _tensor_ok(hidden_states, _device, _in_channels)):
            _record(_call_id, "fallback:signature")
            return _parent(hidden_states, *args, **kwargs)
        out = F.conv2d(hidden_states, _weight, _bias, stride=_conv.stride, padding=_conv.padding,
                       dilation=_conv.dilation, groups=_conv.groups)
        if _activation is not None:
            out = _activation(out)
        _record(_call_id, "folded_conv_generic")
        return out

    return replacement


def _build_shortcut_site(call_id, module, parent, record_route, fold_of, coord):
    conv = module.convolution
    bn = module.normalization
    pair = fold_of(conv, bn)

    def setup_failed(*args, _parent=parent, _call_id=call_id, _record=record_route, **kwargs):
        _record(_call_id, "fallback:fold_setup_failed")
        return _parent(*args, **kwargs)

    if pair is None:
        return setup_failed
    weight, bias = pair
    device = weight.device
    in_channels = int(conv.in_channels)

    def replacement(hidden_states, *args, _parent=parent, _call_id=call_id, _record=record_route,
                    _weight=weight, _bias=bias, _conv=conv, _bn=bn, _coord=coord,
                    _device=device, _in_channels=in_channels, **kwargs):
        if (args or kwargs or torch.is_grad_enabled() or _conv.training or _bn.training
                or not _tensor_ok(hidden_states, _device, _in_channels)):
            _record(_call_id, "fallback:signature")
            return _parent(hidden_states, *args, **kwargs)
        if _coord is not None and _coord.want:
            out = F.conv2d(hidden_states, _weight, None, stride=_conv.stride, padding=_conv.padding,
                           dilation=_conv.dilation, groups=_conv.groups)
            _coord.shortcut_free = True
            _record(_call_id, "coord_shortcut_biasfree")
            return out
        out = F.conv2d(hidden_states, _weight, _bias, stride=_conv.stride, padding=_conv.padding,
                       dilation=_conv.dilation, groups=_conv.groups)
        _record(_call_id, "folded_shortcut_full")
        return out

    return replacement


def _build_basic_layer(call_id, module, parent, record_route, fold_of, coord_for_module):
    conv1 = module.layer[0]
    conv2 = module.layer[1]
    shortcut_mod = module.shortcut
    has_shortcut_conv = _is_shortcut_conv(shortcut_mod)
    conv1_conv = conv1.convolution
    conv1_bn = conv1.normalization
    conv2_conv = conv2.convolution
    conv2_bn = conv2.normalization
    shortcut_conv = shortcut_mod.convolution if has_shortcut_conv else None
    shortcut_bn = shortcut_mod.normalization if has_shortcut_conv else None

    conv1_pair = fold_of(conv1_conv, conv1_bn)
    conv2_pair = fold_of(conv2_conv, conv2_bn)

    def setup_failed(*args, _parent=parent, _call_id=call_id, _record=record_route, **kwargs):
        _record(_call_id, "fallback:fold_setup_failed")
        return _parent(*args, **kwargs)

    if conv1_pair is None or conv2_pair is None:
        return setup_failed

    weight2, bias2 = conv2_pair
    coord = coord_for_module.get(id(conv2))
    if coord is None:
        coord = _Coord()
    shortcut_bias = None
    both_bias = None
    if has_shortcut_conv:
        shortcut_pair = fold_of(shortcut_conv, shortcut_bn)
        if shortcut_pair is not None:
            with torch.no_grad():
                shortcut_bias = shortcut_pair[1]
                both_bias = (bias2 + shortcut_bias).contiguous()
    with torch.no_grad():
        zero_bias = torch.zeros_like(bias2)
    in_channels = int(conv1_conv.in_channels)
    device = weight2.device

    def replacement(hidden_state, *args, _parent=parent, _call_id=call_id, _record=record_route,
                    _layer=module.layer, _shortcut=shortcut_mod, _coord=coord,
                    _zero=zero_bias, _b2=bias2, _bsc=shortcut_bias, _both=both_bias,
                    _has_shortcut_conv=has_shortcut_conv, _device=device, _in_channels=in_channels,
                    _conv1=conv1_conv, _bn1=conv1_bn, _conv2=conv2_conv, _bn2=conv2_bn,
                    _shortcut_conv=shortcut_conv, _shortcut_bn=shortcut_bn, **kwargs):
        if (args or kwargs or torch.is_grad_enabled()
                or _conv1.training or _bn1.training or _conv2.training or _bn2.training
                or (_shortcut_conv is not None and (_shortcut_conv.training or _shortcut_bn.training))
                or not _tensor_ok(hidden_state, _device, _in_channels)):
            _record(_call_id, "fallback:signature")
            return _parent(hidden_state, *args, **kwargs)
        _coord.want = True
        _coord.conv2_free = False
        _coord.shortcut_free = False
        try:
            hidden = _layer(hidden_state)
            residual = _shortcut(hidden_state)
        finally:
            _coord.want = False
        if _coord.conv2_free:
            bias = _b2
        else:
            bias = _zero
        if _has_shortcut_conv and _coord.shortcut_free:
            if _coord.conv2_free and _both is not None:
                bias = _both
            elif _bsc is not None:
                bias = _bsc
        kernel = _RUNTIME_PROBES["triton_kernel"] if not _TRITON_STATE["failed"] else None
        if kernel is not None and hidden.is_contiguous() and residual.is_contiguous():
            try:
                out = _launch_add_bias_relu(hidden, residual, bias, kernel)
                _record(_call_id, "basic_fused_epilogue_triton")
                return out
            except Exception:
                _TRITON_STATE["failed"] = True
                _TRITON_STATE["kernel"] = None
        out = torch.relu(hidden + residual + bias.view(1, -1, 1, 1))
        _record(_call_id, "basic_fused_epilogue_eager")
        return out

    return replacement
