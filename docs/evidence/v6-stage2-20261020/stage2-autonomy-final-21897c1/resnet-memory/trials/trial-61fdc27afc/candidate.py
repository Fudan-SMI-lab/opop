"""Whole-bundle stem replacement for the ResNet-18 "offline" memory project.

The selected region is the entire ``model.resnet.embedder`` bundle
(ResNetEmbeddings = ResNetConvLayer[Conv2d 7x7/s2 bias-free, BatchNorm2d,
ReLU] + MaxPool2d 3/2/1).  The measured baseline peak (282,103,808 B over a
76,710,912 B standing allocation, project.json) is the stem's transient of
205,392,896 B = two simultaneous [32,64,112,112] fp32 buffers
(conv output + out-of-place cudnn BN output, then BN output + out-of-place
relu output; target-resources.json: bn_fw_inf_1C11_kernel 227.7 us,
launch_clamp_scalar relu pass 216.1 us).

This replacement:

  * folds the eval-mode BatchNorm affine (scale = gamma*rsqrt(var+eps),
    shift = beta - mean*scale [+ conv_bias*scale]) into a *private copy* of
    the conv weight/bias built once at install time - the module's own
    parameters and running buffers are never mutated;
  * runs conv + max-pool on batch slices and writes each pooled slice
    straight into one preallocated [B,64,H',W'] output buffer, so the largest
    transient is one slice-sized conv buffer instead of two full stem buffers;
  * commutes the ReLU through the max-pool (exact: clamp_min is monotone
    non-decreasing and max-pool returns one of its inputs, so
    max_pool(relu(t)) == relu(max_pool(t)) elementwise on finite values) and
    applies it while writing each slice, removing the separate out-of-place
    relu buffer and the separate BN kernel entirely.

Every invocation recomputes the whole stem on the current input tensor.  Any
signature/state mismatch (non-fp32/CPU/non-contiguous/wrong rank), any setup
problem, or any runtime error inside the fast path falls back to the parent
forward with a distinct route label; the input tensor and module state are
never mutated, so fallback is exact.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CALL_ID = "66593cf97d10975485c87e23"
DEFAULT_CHUNK = 8


def _folded_stem_params(conv, bn):
    """Return private (weight, bias) copies with the eval-mode BN affine folded in."""
    weight = conv.weight.detach()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    scale = gamma * torch.rsqrt(var + float(bn.eps))
    shift = beta - mean * scale
    if conv.bias is not None:
        shift = shift + conv.bias.detach() * scale
    view = (1,) * (weight.dim() - 1)
    folded_weight = (weight * scale.view(-1, *view)).contiguous()
    return folded_weight, shift.contiguous()


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parameters = context.get("parameters") or {}

    module = modules.get(CALL_ID)
    if module is None:
        raise ValueError(f"module for {CALL_ID} is not present on this rank")
    parent = references.get(CALL_ID)
    if parent is None:
        parent = module.forward

    chunk = parameters.get("stem_chunk", DEFAULT_CHUNK)
    try:
        chunk = max(1, int(chunk))
    except (TypeError, ValueError):
        chunk = DEFAULT_CHUNK

    setup = None
    try:
        conv_layer = module.embedder
        conv = conv_layer.convolution
        bn = conv_layer.normalization
        act = conv_layer.activation
        pool = module.pooler
        if not isinstance(pool, torch.nn.MaxPool2d):
            raise TypeError("stem pooler is not MaxPool2d")
        if pool.return_indices or pool.ceil_mode:
            raise TypeError("unsupported MaxPool2d configuration")
        if not isinstance(act, torch.nn.ReLU):
            raise TypeError("stem activation is not ReLU")
        if conv.padding_mode != "zeros":
            raise TypeError("unsupported conv padding mode")
        if not (getattr(bn, "affine", False) and getattr(bn, "track_running_stats", False)):
            raise TypeError("unsupported BatchNorm2d configuration")
        if bn.running_mean is None or bn.running_var is None:
            raise TypeError("missing BN running stats")
        if bn.num_features != conv.out_channels:
            raise TypeError("BN feature count mismatch")
        # Build the folded copies as normal (non-inference) tensors regardless
        # of the mode the installer happens to run in.
        with torch.inference_mode(False), torch.no_grad():
            folded_weight, folded_bias = _folded_stem_params(conv, bn)
        pool_stride = pool.stride if pool.stride is not None else pool.kernel_size
        setup = {
            "bn": bn,
            "folded_weight": folded_weight,
            "folded_bias": folded_bias,
            "conv_kwargs": {
                "stride": conv.stride,
                "padding": conv.padding,
                "dilation": conv.dilation,
                "groups": conv.groups,
            },
            "pool_args": (pool.kernel_size, pool_stride, pool.padding, pool.dilation, pool.ceil_mode),
            "in_channels": int(conv.in_channels),
        }
    except Exception:
        setup = None  # reported per call as a setup-failure route, never silent

    def replacement(*args, **kwargs):
        if setup is None:
            record_route(CALL_ID, "stem:setup-failure:parent")
            return parent(*args, **kwargs)

        x = args[0] if args else kwargs.get("pixel_values")
        if (
            not torch.is_tensor(x)
            or x.dim() != 4
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.shape[1] != setup["in_channels"]
            or not x.is_contiguous()
            or module.training
            or setup["bn"].training
        ):
            record_route(CALL_ID, "stem:unsupported-signature:parent")
            return parent(*args, **kwargs)

        try:
            fw = setup["folded_weight"]
            fb = setup["folded_bias"]
            ck = setup["conv_kwargs"]
            pk = setup["pool_args"]
            batch = x.shape[0]
            if batch <= chunk:
                t = F.conv2d(x, fw, fb, **ck)
                out = F.max_pool2d(t, *pk)
                out.clamp_min_(0.0)
                record_route(CALL_ID, "stem:folded-single-slice")
                return out
            out = None
            start = 0
            while start < batch:
                stop = min(start + chunk, batch)
                t = F.conv2d(x[start:stop], fw, fb, **ck)
                p = F.max_pool2d(t, *pk)
                if out is None:
                    out = torch.empty(
                        (batch,) + tuple(p.shape)[1:], device=p.device, dtype=p.dtype
                    )
                torch.clamp(p, min=0.0, out=out[start:stop])
                start = stop
            record_route(CALL_ID, "stem:folded-chunked")
            return out
        except Exception:
            record_route(CALL_ID, "stem:runtime-failure:parent")
            return parent(*args, **kwargs)

    return {CALL_ID: replacement}
