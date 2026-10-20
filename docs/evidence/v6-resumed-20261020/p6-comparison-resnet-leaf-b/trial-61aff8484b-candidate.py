from __future__ import annotations

import torch

_CALL_IDS = (
    "417724ae1ac74a4e5cbb794e",
    "c74ed56559b58489b279280e",
    "f9b932ed834da4f5ada556b6",
)

_INPUT_SHAPE = (1, 512, 7, 7)
_INPUT_STRIDE = (25088, 49, 7, 1)
_WEIGHT_SHAPE = (512, 512, 3, 3)
_WEIGHT_STRIDE = (4608, 9, 3, 1)


def _im2col_aten(x):
    return torch.ops.aten.im2col(x, [3, 3], [1, 1], [1, 1], [1, 1])


def _im2col_functional(x):
    return torch.nn.functional.unfold(x, 3, 1, 1, 1)


def _resolve_im2col():
    try:
        probe = _im2col_aten(torch.zeros(1, 1, 2, 2))
        if tuple(probe.shape) == (1, 9, 4):
            return _im2col_aten
    except Exception:
        pass
    return _im2col_functional


def _make_replacement(reference, im2col, matmul):
    def conv2d_leaf(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        if (
            bias is None
            and groups == 1
            and input.is_cuda
            and weight.is_cuda
            and input.dtype is torch.float32
            and weight.dtype is torch.float32
            and input.shape == _INPUT_SHAPE
            and input.stride() == _INPUT_STRIDE
            and weight.shape == _WEIGHT_SHAPE
            and weight.stride() == _WEIGHT_STRIDE
            and list(stride) == [1, 1]
            and list(padding) == [1, 1]
            and list(dilation) == [1, 1]
        ):
            columns = im2col(input)
            output = matmul(weight.view(512, 4608), columns)
            return output.view(1, 512, 7, 7)
        return reference(input, weight, bias, stride, padding, dilation, groups)

    return conv2d_leaf


def build(context):
    im2col = _resolve_im2col()
    matmul = torch.matmul
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    replacements = {}
    for call_id in _CALL_IDS:
        if call_id in targets and call_id in references:
            replacements[call_id] = _make_replacement(references[call_id], im2col, matmul)
    return replacements
