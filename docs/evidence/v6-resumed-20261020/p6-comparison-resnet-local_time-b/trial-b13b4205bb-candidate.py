"""Split-K SGEMM replacement for the three stage-4 aten.conv2d sites of the frozen ResNet-18 workload."""

from __future__ import annotations

import torch

SELECTED_CALLS = (
    "417724ae1ac74a4e5cbb794e",
    "c74ed56559b58489b279280e",
    "f9b932ed834da4f5ada556b6",
)

_IN_SHAPE = (1, 512, 7, 7)
_IN_STRIDE = (25088, 49, 7, 1)
_W_SHAPE = (512, 512, 3, 3)
_W_STRIDE = (4608, 9, 3, 1)
_K = 512 * 3 * 3
_SPATIAL = 7 * 7
_DEFAULT_SPLIT = 12
_ALLOWED_SPLITS = (1, 2, 4, 8, 12, 16, 18, 24, 32, 36)


def _tup(value):
    try:
        return tuple(value)
    except TypeError:
        return value


def _compatible(input, weight, bias, stride, padding, dilation, groups):
    if not (torch.is_tensor(input) and torch.is_tensor(weight)):
        return False
    if input.dim() != 4 or tuple(input.shape) != _IN_SHAPE or tuple(input.stride()) != _IN_STRIDE:
        return False
    if weight.dim() != 4 or tuple(weight.shape) != _W_SHAPE or tuple(weight.stride()) != _W_STRIDE:
        return False
    if input.dtype is not torch.float32 or weight.dtype is not torch.float32:
        return False
    if not input.is_cuda or weight.device != input.device:
        return False
    if bias is not None:
        return False
    if _tup(stride) != (1, 1) or _tup(padding) != (1, 1) or _tup(dilation) != (1, 1):
        return False
    return groups == 1


def _splitk_conv(input, weight, split):
    cols = torch.nn.functional.unfold(input, (3, 3), padding=(1, 1))
    if split == 1:
        y = torch.mm(weight.view(512, _K), cols.view(_K, _SPATIAL))
    else:
        kc = _K // split
        w = weight.view(512, split, kc).permute(1, 0, 2)
        x = cols.view(split, kc, _SPATIAL)
        y = torch.bmm(w, x).sum(dim=0)
    return y.view(1, 512, 7, 7)


def _make_replacement(reference, split):
    def conv2d_splitk(
        input,
        weight,
        bias=None,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        if _compatible(input, weight, bias, stride, padding, dilation, groups):
            try:
                return _splitk_conv(input, weight, split)
            except Exception:
                pass
        if reference is not None:
            return reference(input, weight, bias, stride, padding, dilation, groups)
        return torch.ops.aten.conv2d(input, weight, bias, stride, padding, dilation, groups)

    return conv2d_splitk


def build(context):
    parameters = context.get("parameters") or {}
    split = parameters.get("split_k", _DEFAULT_SPLIT)
    try:
        split = int(split)
    except (TypeError, ValueError):
        split = _DEFAULT_SPLIT
    if split not in _ALLOWED_SPLITS or _K % split != 0:
        split = _DEFAULT_SPLIT

    references = context.get("references") or {}
    targets = context.get("targets") or {}

    replacements = {}
    for call_id in SELECTED_CALLS:
        if call_id not in targets:
            continue
        replacements[call_id] = _make_replacement(references.get(call_id), split)
    return replacements
