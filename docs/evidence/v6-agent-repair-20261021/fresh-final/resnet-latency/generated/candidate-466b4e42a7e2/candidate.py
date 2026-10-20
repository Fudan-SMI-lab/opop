"""Joint BN-folded convolution replacements for the ResNet-18 offline latency project.

The measured unit is one complete FP32 batch=32 forward of the whole ResNet-18
(adapter.run -> ``model(pixel_values=pixels).logits`` under
``torch.inference_mode()``). Each of the 20 selected module sites (17
``transformers...ResNetConvLayer`` and 3 ``ResNetShortCut`` calls; selection
groups 836e80426d566f8f5dce, 0022893ec8c1ceea869f, 39b672d8d4bea8959c44,
867f8d20e37fedce71c4, c30ffc96e594cd4d7734, 04fe5ffd91a3ecdc1ced,
9c1e6f2c47a84d7de11d, 4e7e7ecf95e285fe008e) runs with training=False and a
frozen eval-mode BatchNorm2d directly after a bias-free conv, so the pair is
exactly foldable for inference:

    scale = gamma / sqrt(running_var + eps)
    W'    = W * scale
    b'    = beta + scale * (b0 - running_mean)        (b0 = 0 at every site)

Each replacement issues one ``F.conv2d(x, W', b')`` with the site's original
stride/padding/dilation/groups and then invokes the site's unchanged
activation module (ReLU or Identity; shortcuts have none). All 20
``aten::batch_norm`` kernels disappear while every ``aten::conv2d``,
``aten::relu``, ``max_pool2d`` and ``add_`` kernel still executes on the
current inputs at every invocation. Folded tensors are derived once at build
time from the invariant eval-mode parameters; the original parameters and BN
buffers are left untouched so every guarded fallback re-enters the original
parent forward with unchanged semantics.

Route labels reported through ``context['record_route']``:
  "bn_folded_conv2d"         - folded fast path taken
  "fallback:signature"       - unsupported signature/state, parent used
  "fallback:fold_setup_failed" - fold prerequisites missing, parent used
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# 17 ResNetConvLayer + 3 ResNetShortCut module call IDs (selection.json).
SITE_CALL_IDS = (
    "04de0e7be76279cb340fc51e",
    "1b160dbf6743acd183be0d7d",
    "1b163dbd8c0a67a214a9f004",
    "2cf4d2b82dc58e11738bb3b9",
    "31651bd556566af82ff06f6a",
    "3a8cb56e234373d6905df319",
    "3db154d4b6654cb04cba4425",
    "4e1cb6f216767d70f53f37cb",
    "6eed86b680edffea52f8c0c7",
    "86d53f173ec2db711c715400",
    "9000547cd2a8bfb3ff3c94e8",
    "9337819e1d7b6a87031cc9fa",
    "9c61a0898adb09fcada55871",
    "ab5f018984612357036fff22",
    "ad0730b4f031b770865aca4c",
    "cecf17ef310a4487f93c1329",
    "d97d55bb22b7a5b14da54326",
    "db8a4685e26c9341b26c8e7a",
    "f9ad704797a591fd3fd254a0",
    "fad32b28bd6e0def6ce7593e",
)


def _fold_conv_bn(conv, bn):
    """Exact eval-mode BN fold. Returns (W', b') tensors or None if not foldable."""
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


def build(context):
    modules = context.get("modules", {})
    references = context.get("references", {})
    record_route = context.get("record_route", lambda *_: None)

    replacements = {}
    for call_id in SITE_CALL_IDS:
        module = modules.get(call_id)
        parent = references.get(call_id)
        if module is None or parent is None:
            continue

        conv = getattr(module, "convolution", None)
        bn = getattr(module, "normalization", None)
        try:
            folded = _fold_conv_bn(conv, bn) if conv is not None and bn is not None else None
        except Exception:
            folded = None

        if folded is None:
            def setup_failed(*args, _parent=parent, _call_id=call_id, _record=record_route, **kwargs):
                _record(_call_id, "fallback:fold_setup_failed")
                return _parent(*args, **kwargs)

            replacements[call_id] = setup_failed
            continue

        weight, bias = folded
        activation = getattr(module, "activation", None)

        def replacement(
            hidden_states,
            *args,
            _parent=parent,
            _call_id=call_id,
            _record=record_route,
            _weight=weight,
            _bias=bias,
            _activation=activation,
            _stride=conv.stride,
            _padding=conv.padding,
            _dilation=conv.dilation,
            _groups=conv.groups,
            _in_channels=conv.in_channels,
            _device=weight.device,
            _conv=conv,
            _bn=bn,
            **kwargs,
        ):
            if (
                args
                or kwargs
                or _conv.training
                or _bn.training
                or torch.is_grad_enabled()
                or not isinstance(hidden_states, torch.Tensor)
                or hidden_states.dim() != 4
                or hidden_states.dtype is not torch.float32
                or hidden_states.device != _device
                or hidden_states.shape[1] != _in_channels
            ):
                _record(_call_id, "fallback:signature")
                return _parent(hidden_states, *args, **kwargs)
            out = F.conv2d(
                hidden_states,
                _weight,
                _bias,
                stride=_stride,
                padding=_padding,
                dilation=_dilation,
                groups=_groups,
            )
            if _activation is not None:
                out = _activation(out)
            _record(_call_id, "bn_folded_conv2d")
            return out

        replacements[call_id] = replacement

    return replacements
