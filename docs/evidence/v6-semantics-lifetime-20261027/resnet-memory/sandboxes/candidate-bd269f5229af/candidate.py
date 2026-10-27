"""Memory-lean composite for the observed ResNet-18 stem bundle (model.resnet.embedder).

The measured peak_allocated_bytes moment of the frozen offline batch-32 forward is
the eager stem: during aten::batch_norm the conv1 output (32x64x112x112 fp32,
102,760,448 B) and the BN output (another 102,760,448 B) are simultaneously live on
top of resident weights and the GPU-resident input. conv2d, eval-mode batch_norm,
relu and max_pool2d are all strictly per-sample operators, so streaming the batch
through the stem in fixed per-sample chunks and rejoining before stage 0 removes the
full-resolution temporaries without changing any operator, dtype, tolerance-relevant
arithmetic or workload semantics. Every chunk still executes the observed parent
aten operators through context['references'], so the parent work keeps running on
each invocation and reference telemetry records it.
"""

from __future__ import annotations

import torch
from torch.nn.modules.utils import _pair

STEM = "fa7ef6548e3a2d54174caeba"
CONV_MODULE = "a929588cc0791799075542c6"
NORM_MODULE = "d2690ee924ab33cd95efa6f0"
POOL_MODULE = "12da68be9b9e14502fe4702a"
CONV_OP = "8504783bbca736066fecb204"
NORM_OP = "89bc7485bcb9aeaa62c7d28b"
RELU_OP = "89f968d79a78e013c08bd56c"
POOL_OP = "4367176c34e4141fee648ee6"

CHUNK_SAMPLES = 8


def build(context):
    references = context.get("references", {}) or {}
    modules = context.get("modules", {}) or {}
    record_route = context.get("record_route", lambda *_: None)

    parent = references.get(STEM)
    conv = modules.get(CONV_MODULE)
    norm = modules.get(NORM_MODULE)
    pool = modules.get(POOL_MODULE)
    conv_op = references.get(CONV_OP)
    norm_op = references.get(NORM_OP)
    relu_op = references.get(RELU_OP)
    pool_op = references.get(POOL_OP)
    ready = all(
        item is not None
        for item in (parent, conv, norm, pool, conv_op, norm_op, relu_op, pool_op)
    )

    def supported(x):
        if not ready:
            return False
        try:
            return bool(
                isinstance(x, torch.Tensor)
                and x.is_cuda
                and x.dim() == 4
                and x.dtype == torch.float32
                and x.shape[0] > 0
                and x.shape[1] == int(conv.in_channels)
                and conv.weight is not None
                and conv.weight.dtype == torch.float32
                and conv.weight.is_cuda
                and not bool(conv.training)
                and not bool(norm.training)
                and bool(norm.affine)
                and bool(norm.track_running_stats)
                and norm.weight is not None
                and norm.bias is not None
                and norm.running_mean is not None
                and norm.running_var is not None
                and not bool(pool.return_indices)
                and not bool(pool.ceil_mode)
                and str(getattr(conv, "padding_mode", "zeros")) == "zeros"
                and int(getattr(conv, "groups", 1)) == 1
                and int(norm.num_features) == int(conv.out_channels)
            )
        except Exception:
            return False

    def stem(*args, **kwargs):
        x = args[0] if args else kwargs.get("pixel_values")
        if not supported(x):
            record_route(STEM, "setup_failure_parent" if not ready else "unsupported_signature_parent")
            return parent(*args, **kwargs)
        try:
            record_route(STEM, "chunked_stem_composite")
            weight = conv.weight
            stride = _pair(conv.stride)
            padding = _pair(conv.padding)
            dilation = _pair(conv.dilation)
            groups = int(conv.groups)
            bn_weight = norm.weight
            bn_bias = norm.bias
            running_mean = norm.running_mean
            running_var = norm.running_var
            momentum = float(norm.momentum)
            eps = float(norm.eps)
            kernel_size = _pair(pool.kernel_size)
            pool_stride = _pair(pool.stride)
            pool_padding = _pair(pool.padding)
            pool_dilation = _pair(pool.dilation)
            total = int(x.shape[0])
            step = CHUNK_SAMPLES if total >= CHUNK_SAMPLES else total
            out = None
            for start in range(0, total, step):
                stop = min(start + step, total)
                piece = x[start:stop]
                h = conv_op(piece, weight, conv.bias, stride, padding, dilation, groups)
                h = norm_op(h, bn_weight, bn_bias, running_mean, running_var, False, momentum, eps, True)
                h = relu_op(h)
                h = pool_op(h, kernel_size, pool_stride, pool_padding, pool_dilation, False)
                if out is None:
                    out = torch.empty((total, *h.shape[1:]), dtype=h.dtype, device=h.device)
                out[start:stop].copy_(h)
            return out
        except Exception:
            record_route(STEM, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return {STEM: stem}
