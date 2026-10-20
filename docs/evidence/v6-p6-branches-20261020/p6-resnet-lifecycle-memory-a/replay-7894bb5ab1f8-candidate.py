"""Composite memory candidate for the frozen ResNet-18 offline (batch=32, FP32, NCHW) workload.

Folds the eval-mode stem BatchNorm (model.resnet.embedder.embedder.normalization)
into the 7x7 stride-2 stem convolution (model.resnet.embedder.embedder.convolution)
once during build, and applies the stem ReLU in place on the folded conv output,
so the embeddings stage keeps exactly one (batch,64,112,112) FP32 activation
buffer live instead of the baseline conv-output/BN-output pair.
"""
from __future__ import annotations

import torch

CONV2D_CALL = "220b9d0ec76e2ccc01dcdb09"
BATCHNORM_CALL = "279a256c150755fd69216f5a"
RELU_CALL = "ddf82d5c5c82c55ac9ae09f1"
CONV_MODULE_CALL = "8ef2341287e5b4afa9d0e7bf"
BN_MODULE_CALL = "2f3bfb9ee01d701be01d2ff3"

_STEM_TAG = "_opop_resnet18_stem_folded"


def _pair(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def _find_stem(context):
    modules = context.get("modules") or {}
    conv = modules.get(CONV_MODULE_CALL)
    bn = modules.get(BN_MODULE_CALL)
    if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
        return conv, bn
    for model in (context.get("models") or {}).values():
        node = model
        for part in ("resnet", "embedder", "embedder"):
            nxt = getattr(node, part, None) if isinstance(node, torch.nn.Module) else None
            if not isinstance(nxt, torch.nn.Module):
                node = None
                break
            node = nxt
        if node is not None:
            conv = getattr(node, "convolution", None)
            bn = getattr(node, "normalization", None)
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                return conv, bn
    return None, None


def build(context):
    replacements: dict = {}
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    for call_id in (CONV2D_CALL, BATCHNORM_CALL, RELU_CALL):
        if call_id not in targets:
            return replacements
    conv_ref = references.get(CONV2D_CALL)
    bn_ref = references.get(BATCHNORM_CALL)
    relu_ref = references.get(RELU_CALL)
    if conv_ref is None or bn_ref is None or relu_ref is None:
        return replacements
    conv, bn = _find_stem(context)
    if conv is None or bn is None:
        return replacements
    weight = conv.weight
    bn_weight = bn.weight
    bn_bias = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    if weight is None or bn_weight is None or bn_bias is None:
        return replacements
    if running_mean is None or running_var is None:
        return replacements
    if not bool(bn.track_running_stats) or bool(bn.training) or bool(conv.training):
        return replacements
    channels = int(weight.shape[0]) if weight.dim() == 4 else -1
    if channels <= 0 or int(bn.num_features) != channels:
        return replacements
    if weight.dtype is not torch.float32 or bn_weight.dtype is not torch.float32:
        return replacements
    if tuple(bn_weight.shape) != (channels,) or tuple(bn_bias.shape) != (channels,):
        return replacements
    if tuple(running_mean.shape) != (channels,) or tuple(running_var.shape) != (channels,):
        return replacements
    eps = float(bn.eps)
    stride = _pair(conv.stride)
    padding = _pair(conv.padding)
    dilation = _pair(conv.dilation)
    groups = int(conv.groups)
    in_channels = int(conv.in_channels)
    with torch.no_grad():
        scale = bn_weight / torch.sqrt(running_var + eps)
        folded_weight = (weight * scale.view(-1, 1, 1, 1)).contiguous()
        folded_bias = (bn_bias - running_mean * scale).contiguous()
    try:
        probe = torch.empty(1, device=weight.device)
        setattr(probe, _STEM_TAG, True)
        del probe
        can_tag = True
    except Exception:
        can_tag = False
    id_marks: dict = {}

    def _mark(tensor):
        if can_tag:
            setattr(tensor, _STEM_TAG, True)
        else:
            id_marks[id(tensor)] = True

    def _marked(tensor):
        if can_tag:
            return getattr(tensor, _STEM_TAG, False) is True
        return id_marks.get(id(tensor), False)

    def _unmark(tensor):
        if can_tag:
            try:
                if getattr(tensor, _STEM_TAG, False) is True:
                    delattr(tensor, _STEM_TAG)
            except (AttributeError, RuntimeError):
                pass
        else:
            id_marks.pop(id(tensor), None)

    def conv2d_replacement(*args, **kwargs):
        if kwargs or not 5 <= len(args) <= 7:
            return conv_ref(*args, **kwargs)
        if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
            return conv_ref(*args, **kwargs)
        inp, w, bias = args[0], args[1], args[2]
        call_stride = _pair(args[3])
        call_padding = _pair(args[4])
        call_dilation = _pair(args[5]) if len(args) > 5 else (1, 1)
        call_groups = int(args[6]) if len(args) > 6 else 1
        same_weight = w is weight or (
            w.shape == weight.shape and w.dtype is torch.float32 and bool(torch.equal(w, weight))
        )
        if (
            same_weight
            and bias is None
            and not bn.training
            and inp.dtype is torch.float32
            and inp.device == weight.device
            and inp.dim() == 4
            and int(inp.shape[1]) == in_channels
            and call_stride == stride
            and call_padding == padding
            and call_dilation == dilation
            and call_groups == groups
        ):
            out = torch.conv2d(inp, folded_weight, folded_bias, stride, padding, dilation, groups)
            _mark(out)
            return out
        return conv_ref(*args, **kwargs)

    def batchnorm_replacement(*args, **kwargs):
        if kwargs or len(args) != 9 or not isinstance(args[0], torch.Tensor):
            return bn_ref(*args, **kwargs)
        inp = args[0]
        if _marked(inp) and inp.dtype is torch.float32:
            return inp
        return bn_ref(*args, **kwargs)

    def relu_replacement(*args, **kwargs):
        if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
            return relu_ref(*args, **kwargs)
        inp = args[0]
        if _marked(inp) and inp.dtype is torch.float32 and inp.device == weight.device:
            out = inp.relu_()
            _unmark(inp)
            return out
        return relu_ref(*args, **kwargs)

    replacements[CONV2D_CALL] = conv2d_replacement
    replacements[BATCHNORM_CALL] = batchnorm_replacement
    replacements[RELU_CALL] = relu_replacement
    return replacements
