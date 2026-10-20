"""Whole-region composite replacement for the frozen ResNet-18 classifier forward.

Region: group eca3643289778aea5af3 -> module call f2cf62caf57ad09f5d2f36aa
(model = transformers ResNetForImageClassification, eval mode, FP32, CUDA).

Build time (installation):
  * verify the module tree matches the observed HF ResNet structure and that
    every normalization is an eval-mode affine BatchNorm2d directly consuming a
    bias-free Conv2d;
  * fold each BatchNorm into its preceding Conv2d with a correctly rounded
    float64 intermediate:
        scale    = gamma / sqrt(running_var + eps)
        weight'  = weight  * scale            (per output channel)
        bias'    = beta - running_mean * scale
  * keep the folded tensors as fresh plain tensors; the original parameters,
    buffers and modules are never mutated, so the captured parent forward stays
    exactly valid as a fallback.

Run time (replacement callable, parent forward ABI without self):
  * replay the identical op sequence conv/relu/maxpool/conv/relu/conv/add_/relu
    /adaptive_avg_pool2d/flatten/linear with the folded weights and biases;
  * relu is applied in place on freshly allocated conv outputs and the residual
    merge keeps the parent's in-place add_; both are observably equivalent
    (no external aliasing: every rewritten tensor is an internal intermediate
    created inside this composite);
  * return the parent's exact result type (ImageClassifierOutputWithNoAttention
    with loss=None, logits=..., hidden_states=None);
  * any call that does not match the observed signature class (positional or
    keyword pixel_values only, other parent kwargs all None, float32, 4-D, on
    the model device, channel dim == num_channels) or any state change (module
    switched to train mode, parameters replaced, shortcut/config structure
    changed) routes back to the captured parent forward and is reported through
    context['record_route']; a concise setup-failure route is kept for the
    unexpected-structure case instead of silently dropping the replacement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

CALL_ID = "f2cf62caf57ad09f5d2f36aa"

_ROUTE_FAST = "folded_bn_composite"
_ROUTE_FALLBACK = "parent_signature_or_state_fallback"
_ROUTE_SETUP_FAILURE = "setup_failure_parent_forward"

_FP32 = torch.float32


def _fold(conv, bn):
    """Fold an eval-mode affine BatchNorm2d into a bias-free Conv2d."""
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("unexpected conv/bn module types")
    if bn.training:
        raise ValueError("batch_norm not in eval mode")
    if conv.bias is not None:
        raise ValueError("conv2d already carries a bias")
    if not bn.affine or bn.weight is None or bn.bias is None:
        raise ValueError("non-affine batch_norm")
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("batch_norm without running stats")
    if bn.num_features != conv.out_channels:
        raise ValueError("channel mismatch between conv and batch_norm")
    if conv.groups != 1 or tuple(conv.dilation) != (1, 1):
        raise ValueError("grouped or dilated conv not supported")
    w, gamma, beta = conv.weight, bn.weight, bn.bias
    mean, var = bn.running_mean, bn.running_var
    if w.dtype is not _FP32 or gamma.dtype is not _FP32 or mean.dtype is not _FP32:
        raise ValueError("non-float32 conv/bn parameters")
    eps = float(bn.eps)
    with torch.no_grad():
        inv_std = torch.rsqrt(var.double() + eps)
        scale = gamma.double() * inv_std
        broadcast = (1,) * (w.dim() - 1)
        weight_f = (w.double() * scale.view(-1, *broadcast)).to(_FP32).contiguous()
        bias_f = (beta.double() - mean.double() * scale).to(_FP32).contiguous()
    return {
        "w": weight_f,
        "b": bias_f,
        "stride": conv.stride,
        "padding": conv.padding,
        "dilation": conv.dilation,
        "groups": conv.groups,
        "conv": conv,
        "wref": conv.weight,
        "bn": bn,
    }


def _fold_conv_block_layer(sub):
    """Fold one ResNetConvLayer-like module (conv + bn [+ activation])."""
    conv = getattr(sub, "convolution", None)
    bn = getattr(sub, "normalization", None)
    act = getattr(sub, "activation", None)
    fold = _fold(conv, bn)
    if act is None or isinstance(act, nn.Identity):
        return fold, False
    if not isinstance(act, nn.ReLU):
        raise TypeError("unsupported activation")
    return fold, True


def _build_plan(model):
    """Verify structure and precompute the folded execution plan.

    Returns a plan dict, or None when the tree does not match the observed
    ResNetForImageClassification semantics (caller keeps a setup-failure shim).
    """
    import transformers.models.resnet.modeling_resnet as modeling_resnet

    if not isinstance(model, modeling_resnet.ResNetForImageClassification):
        return None
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if getattr(cfg, "return_dict", True) is not True:
        return None
    if getattr(cfg, "output_hidden_states", False):
        return None
    if int(getattr(cfg, "num_labels", 0)) <= 0:
        return None
    if model.training:
        return None

    resnet = getattr(model, "resnet", None)
    emb = getattr(resnet, "embedder", None)
    encoder = getattr(resnet, "encoder", None)
    avg_pool = getattr(resnet, "pooler", None)
    classifier = getattr(model, "classifier", None)
    if resnet is None or emb is None or encoder is None or avg_pool is None:
        return None
    if not isinstance(avg_pool, nn.AdaptiveAvgPool2d):
        return None
    if not isinstance(classifier, nn.Sequential) or len(classifier) != 2:
        return None
    flatten, linear = classifier[0], classifier[1]
    if not isinstance(flatten, nn.Flatten) or flatten.start_dim != 1 or flatten.end_dim != -1:
        return None
    if not isinstance(linear, nn.Linear) or linear.bias is None:
        return None
    if linear.weight.dtype is not _FP32:
        return None

    # Stem: ResNetConvLayer(7x7 s2) + MaxPool2d(3, s2, p1).
    stem = getattr(emb, "embedder", None)
    max_pool = getattr(emb, "pooler", None)
    num_channels = getattr(emb, "num_channels", None)
    if stem is None or not isinstance(max_pool, nn.MaxPool2d):
        return None
    if not isinstance(num_channels, int) or num_channels <= 0:
        return None
    if max_pool.return_indices:
        return None
    stem_fold, stem_has_act = _fold_conv_block_layer(stem)
    if not stem_has_act:
        return None
    pool_args = (
        max_pool.kernel_size,
        max_pool.stride if max_pool.stride is not None else max_pool.kernel_size,
        max_pool.padding,
        max_pool.dilation,
        bool(max_pool.ceil_mode),
    )

    # Encoder stages: ResNetBasicLayer blocks (conv[+act] x n, shortcut, add, post-act).
    stages = []
    identity_shortcuts = []
    for stage in encoder.stages:
        blocks = []
        for block in stage.layers:
            convs = []
            acts = []
            for sub in block.layer:
                fold, has_act = _fold_conv_block_layer(sub)
                convs.append(fold)
                acts.append(has_act)
            if not convs:
                return None
            shortcut = getattr(block, "shortcut", None)
            if shortcut is None or isinstance(shortcut, nn.Identity):
                identity_shortcuts.append((block, shortcut))
                shortcut_fold = None
            else:
                sc_conv = getattr(shortcut, "convolution", None)
                sc_bn = getattr(shortcut, "normalization", None)
                if getattr(shortcut, "activation", None) is not None:
                    return None
                shortcut_fold = _fold(sc_conv, sc_bn)
            post = getattr(block, "activation", None)
            if not isinstance(post, nn.ReLU):
                return None
            blocks.append({"convs": convs, "acts": acts, "shortcut": shortcut_fold})
        stages.append(blocks)

    conv_wrefs = [(f["conv"], f["wref"]) for stage in stages for blk in stage for f in blk["convs"]]
    conv_wrefs.append((stem_fold["conv"], stem_fold["wref"]))
    for blk in (b for stage in stages for b in stage):
        if blk["shortcut"] is not None:
            conv_wrefs.append((blk["shortcut"]["conv"], blk["shortcut"]["wref"]))
    bns = [f["bn"] for stage in stages for blk in stage for f in blk["convs"]]
    bns.append(stem_fold["bn"])
    for blk in (b for stage in stages for b in stage):
        if blk["shortcut"] is not None:
            bns.append(blk["shortcut"]["bn"])

    return {
        "output_type": modeling_resnet.ImageClassifierOutputWithNoAttention,
        "device": stem_fold["w"].device,
        "num_channels": num_channels,
        "stem": stem_fold,
        "pool_args": pool_args,
        "stages": stages,
        "avg_pool_output_size": avg_pool.output_size,
        "linear_weight": linear.weight,
        "linear_bias": linear.bias,
        "linear_ref": (linear, linear.weight, linear.bias),
        "model": model,
        "bns": bns,
        "conv_wrefs": conv_wrefs,
        "identity_shortcuts": identity_shortcuts,
    }


def _match_input(args, kwargs):
    """Extract pixel_values for call shapes equivalent to the observed one."""
    n = len(args)
    keys = set(kwargs)
    optional = ("labels", "output_hidden_states", "return_dict")
    if n == 1 and not keys:
        return args[0], True
    if n == 0 and keys == {"pixel_values"}:
        return kwargs["pixel_values"], True
    if (
        n == 0
        and "pixel_values" in keys
        and keys <= {"pixel_values", *optional}
        and all(kwargs.get(k) is None for k in optional)
    ):
        return kwargs["pixel_values"], True
    return None, False


def _run_plan(plan, pixel_values):
    conv2d = F.conv2d
    relu = F.relu

    stem = plan["stem"]
    x = conv2d(
        pixel_values,
        stem["w"],
        stem["b"],
        stride=stem["stride"],
        padding=stem["padding"],
        dilation=stem["dilation"],
        groups=stem["groups"],
    )
    x = relu(x, inplace=True)
    k, s, p, d, ceil = plan["pool_args"]
    x = F.max_pool2d(x, k, stride=s, padding=p, dilation=d, ceil_mode=ceil)

    for stage in plan["stages"]:
        for block in stage:
            residual = x
            folds = block["convs"]
            acts = block["acts"]
            for i, fold in enumerate(folds):
                x = conv2d(
                    x,
                    fold["w"],
                    fold["b"],
                    stride=fold["stride"],
                    padding=fold["padding"],
                    dilation=fold["dilation"],
                    groups=fold["groups"],
                )
                if acts[i]:
                    x = relu(x, inplace=True)
            shortcut = block["shortcut"]
            if shortcut is not None:
                residual = conv2d(
                    residual,
                    shortcut["w"],
                    shortcut["b"],
                    stride=shortcut["stride"],
                    padding=shortcut["padding"],
                    dilation=shortcut["dilation"],
                    groups=shortcut["groups"],
                )
            x = x.add_(residual)
            x = relu(x, inplace=True)

    x = F.adaptive_avg_pool2d(x, plan["avg_pool_output_size"])
    x = torch.flatten(x, 1)
    return F.linear(x, plan["linear_weight"], plan["linear_bias"])


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parent = references.get(CALL_ID)
    model = modules.get(CALL_ID)

    if not callable(parent):
        # No parent reference captured: install nothing, framework keeps parent.
        return {}

    def _setup_failure_forward(*args, **kwargs):
        record_route(CALL_ID, _ROUTE_SETUP_FAILURE)
        return parent(*args, **kwargs)

    try:
        plan = _build_plan(model) if isinstance(model, nn.Module) else None
    except Exception:
        plan = None
    if plan is None:
        return {CALL_ID: _setup_failure_forward}

    device = plan["device"]
    num_channels = plan["num_channels"]
    bns = plan["bns"]
    conv_wrefs = plan["conv_wrefs"]
    identity_shortcuts = plan["identity_shortcuts"]
    linear_ref = plan["linear_ref"]
    root_model = plan["model"]

    def _state_intact():
        if root_model.training:
            return False
        for bn in bns:
            if bn.training:
                return False
        for conv, wref in conv_wrefs:
            if conv.weight is not wref:
                return False
        for block, shortcut in identity_shortcuts:
            if block.shortcut is not shortcut or not isinstance(shortcut, nn.Identity):
                return False
        linear, weight_ref, bias_ref = linear_ref
        if linear.weight is not weight_ref or linear.bias is not bias_ref:
            return False
        return True

    def _forward(*args, **kwargs):
        pixel_values, matched = _match_input(args, kwargs)
        if not matched:
            record_route(CALL_ID, _ROUTE_FALLBACK)
            return parent(*args, **kwargs)
        if not (
            torch.is_tensor(pixel_values)
            and pixel_values.dim() == 4
            and pixel_values.device == device
            and pixel_values.dtype is _FP32
            and pixel_values.shape[1] == num_channels
            and _state_intact()
        ):
            # Delegate so the parent produces the identical result or the
            # identical error (e.g. its channel-dimension ValueError).
            record_route(CALL_ID, _ROUTE_FALLBACK)
            return parent(*args, **kwargs)
        record_route(CALL_ID, _ROUTE_FAST)
        logits = _run_plan(plan, pixel_values)
        return plan["output_type"](loss=None, logits=logits, hidden_states=None)

    return {CALL_ID: _forward}
