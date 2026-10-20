"""Group candidate for the frozen ResNet-18 FP32 batch=1 online inference workload.

Selected group 9a28856391530afa25fa: the three stage-4 aten::conv2d.default
invocations (input 1x512x7x7 contiguous, weight 512x512x3x3, bias 512, stride
(1,1), padding (1,1), dilation (1,1), groups 1).  Per target-resources each
observed call spends ~69us GPU as generateWinogradTilesKernel (11.3-11.5us)
plus ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148n_nt_v1 launched
with grid [16,1,1] (56.0-56.4us) plus a 1.5us elementwise bias-add epilogue.

The replacement computes the identical linear map as explicit im2col plus one
FP32 GEMM with fused bias broadcast:

    cols = F.unfold(x, (3, 3), padding=(1, 1))                        # [1, 4608, 49]
    out  = torch.addmm(bias.view(512, 1), W.view(512, 4608), cols[0]) # [512, 49]
    y    = out.reshape(1, 512, 7, 7)

which removes the winograd tile-generation pass, exchanges the observed
winograd kernel for a cuBLAS SGEMM that can draw parallelism from split-K over
K=4608, and folds the bias epilogue into the GEMM.  The numerical policy is
unchanged: FP32 tensors, FP32 accumulation, TF32 disabled for both matmul and
cudnn, float32_matmul_precision 'highest'; only the accumulation order is
reassociated relative to the winograd formulation.

The packed weight is a zero-copy view of the observed contiguous weight, built
once per physical call site and cached keyed by (data_ptr, _version) so any
weight mutation repacks.  Any invocation whose arguments do not exactly match
the recorded shape/stride/dtype/device and scalar signature falls back to the
parent torch operator with the original arguments untouched, so unobserved
inputs keep exact parent semantics.  Every output is a fresh contiguous NCHW
FP32 tensor with the layout aten::conv2d produces for these inputs; nothing is
mutated in place and no output buffers are retained between calls.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SELECTED_CALLS = (
    "20751897973c79a00ee9fdca",
    "958f42dd68b163e99a8ffc57",
    "f1efdd73d2cc1202e0d29d0a",
)

_ARG_NAMES = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
_ARG_DEFAULTS = (None, None, None, (1, 1), (0, 0), (1, 1), 1)


def _expected_dtype(name):
    if not isinstance(name, str) or not name.startswith("torch."):
        return None
    return getattr(torch, name[len("torch."):], None)


def _tensor_matches(tensor, spec):
    if not isinstance(tensor, torch.Tensor):
        return False
    shape = spec.get("shape")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        return False
    stride = spec.get("stride")
    if stride is not None and tensor.stride() != tuple(stride):
        return False
    dtype = _expected_dtype(spec.get("dtype"))
    if dtype is not None and tensor.dtype is not dtype:
        return False
    if spec.get("device") == "cuda" and not tensor.is_cuda:
        return False
    return True


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (value[0], value[1])
    return None


def _normalize(args, kwargs):
    if len(args) > len(_ARG_NAMES):
        return None
    names_tail = _ARG_NAMES[len(args):]
    if any(name not in names_tail for name in kwargs):
        return None
    values = list(args)
    for offset, name in enumerate(names_tail):
        if name in kwargs:
            values.append(kwargs[name])
        else:
            values.append(_ARG_DEFAULTS[len(args) + offset])
    return values


def _make_replacement(input_spec, weight_spec, bias_spec, geometry, reference):
    batch, out_channels, out_h, out_w, kernel, stride, padding, dilation = geometry
    unfold = F.unfold
    addmm = torch.addmm
    cache = {}

    def packed_weight(weight):
        key = (weight.data_ptr(), weight._version)
        packed = cache.get(key)
        if packed is None:
            packed = weight.reshape(out_channels, -1)
            if len(cache) >= 4:
                cache.clear()
            cache[key] = packed
        return packed

    def replacement(*args, **kwargs):
        values = _normalize(args, kwargs)
        if values is None:
            return reference(*args, **kwargs)
        x, w, b, s_stride, s_padding, s_dilation, s_groups = values
        if s_groups != 1:
            return reference(*args, **kwargs)
        if _pair(s_stride) != stride or _pair(s_padding) != padding or _pair(s_dilation) != dilation:
            return reference(*args, **kwargs)
        if not (
            _tensor_matches(x, input_spec)
            and _tensor_matches(w, weight_spec)
            and _tensor_matches(b, bias_spec)
        ):
            return reference(*args, **kwargs)
        if w.device != x.device or b.device != x.device:
            return reference(*args, **kwargs)
        cols = unfold(x, kernel, dilation=dilation, padding=padding, stride=stride)
        out = addmm(b.view(-1, 1), packed_weight(w), cols[0])
        return out.reshape(batch, out_channels, out_h, out_w)

    return replacement


def build(context):
    replacements = {}
    if not isinstance(context, dict):
        return replacements
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    for call_id in _SELECTED_CALLS:
        spec = targets.get(call_id)
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") != "functional" or spec.get("target") != "aten.conv2d.default":
            continue
        signature = (spec.get("signatures") or [None])[0]
        if not isinstance(signature, dict):
            continue
        args_spec = signature.get("args") or []
        if len(args_spec) < 5:
            continue
        input_spec, weight_spec, bias_spec = args_spec[0], args_spec[1], args_spec[2]
        if not all(isinstance(item, dict) for item in (input_spec, weight_spec, bias_spec)):
            continue
        stride = _pair(args_spec[3])
        padding = _pair(args_spec[4])
        if stride is None or padding is None:
            continue
        x_shape = input_spec.get("shape")
        w_shape = weight_spec.get("shape")
        if not x_shape or not w_shape or len(x_shape) != 4 or len(w_shape) != 4:
            continue
        batch, in_channels, in_h, in_w = (int(v) for v in x_shape)
        out_channels, w_in, k_h, k_w = (int(v) for v in w_shape)
        if w_in != in_channels or min(batch, out_channels, k_h, k_w) < 1:
            continue
        dilation = (1, 1)
        out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
        if out_h < 1 or out_w < 1:
            continue
        geometry = (batch, out_channels, out_h, out_w, (k_h, k_w), stride, padding, dilation)
        reference = references.get(call_id)
        if not callable(reference):
            reference = torch.conv2d
        replacements[call_id] = _make_replacement(
            input_spec, weight_spec, bias_spec, geometry, reference
        )
    return replacements
