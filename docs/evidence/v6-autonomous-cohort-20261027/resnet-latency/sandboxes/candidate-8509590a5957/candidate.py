"""Joint candidate for the ResNet-18 offline batch-32 latency task (revision 3bd4af68).

Replaces the earlier accepted BN-fold bundle ``trial-a7f5de7ac7-param-23c435d4``
(17 ``ResNetConvLayer`` sites) as a whole and additionally owns the residual
block epilogue, at 12 module boundaries:

  * ``ResNetEmbeddings`` stem site (1): the 7x7 stem conv with its eval BN
    folded runs as one ``aten.cudnn_convolution_relu`` kernel on packed
    weights, followed by the parent ``MaxPool2d`` module call (unchanged);
  * ``ResNetBasicLayer`` sites (8): layer[0] runs the accepted folded
    ``aten.cudnn_convolution_relu``; the residual is produced by calling
    ``self.shortcut`` exactly like the parent (Identity shortcuts keep their
    parent module call, the 3 ``ResNetShortCut`` sites run this bundle's
    folded replacement); layer[1] with the residual add and the final ReLU
    becomes one ``aten.cudnn_convolution_add_relu`` kernel
    (relu(conv(x, W', b') + 1.0 * z)) on folded weights;
  * ``ResNetShortCut`` sites (3): eval BN folded into the 1x1 conv weights,
    emitted as one conv+bias kernel via ``conv2d`` (the accepted fold).

All folded weights are packed once per physical conv module at build time in
float64 and stored as contiguous float32 (invariant model parameters; the
original parameter tensors are untouched, so every parent fallback recomputes
the exact original parent computation).  Per-site fallbacks preserve the
parent semantics:

  * setup probe declines or runtime failure of a fused op -> the kernel-for-
    kernel parent-equivalent sequence (``conv2d``(W', b') [+ ``add_``] +
    ``relu``), latched per site;
  * unexpected signature/dtype/device/training/grad context -> parent forward.

No global functional dispatch is installed (module-boundary callables only),
no CUDA graphs are captured and no outputs are cached: every invocation
executes the workload on its current inputs.  Probes use deterministic
``linspace`` tensors (no RNG consumption) and are synchronized so setup-level
fused-op failures surface at installation, not during the invocation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_CUDNN_CONV_RELU = torch.ops.aten.cudnn_convolution_relu
_CUDNN_CONV_ADD_RELU = torch.ops.aten.cudnn_convolution_add_relu
_EPILOGUE_ROUTES = ("cudnn_conv_add_relu", "conv2d_add_relu")


def _pack_bn_into_conv(conv, bn):
    """Return (weight', bias') with eval BatchNorm folded into the conv.

    Folding is done in float64 and cast to float32 so the packed tensors are
    the closest float32 representation of the exact transform
      y = ((W*x + c) - mu) * gamma/sqrt(var+eps) + beta
        = (W*s)*x + (beta - mu*s + c*s).
    Returns None when the module layout is not supported.
    """
    if (
        not isinstance(conv, torch.nn.Conv2d)
        or not isinstance(bn, torch.nn.BatchNorm2d)
        or bn.affine is not True
        or bn.track_running_stats is not True
        or bn.weight is None
        or bn.bias is None
        or bn.running_mean is None
        or bn.running_var is None
    ):
        return None
    weight = conv.weight
    if weight is None or weight.dtype is not torch.float32 or not weight.is_cuda:
        return None
    with torch.no_grad():
        eps = float(bn.eps)
        scale = bn.weight.double() / (bn.running_var.double() + eps).sqrt()
        packed_weight = (weight.double() * scale.reshape(-1, 1, 1, 1)).to(torch.float32).contiguous()
        bias64 = bn.bias.double() - bn.running_mean.double() * scale
        if conv.bias is not None:
            bias64 = bias64 + conv.bias.double() * scale
        packed_bias = bias64.to(torch.float32).contiguous()
    return packed_weight, packed_bias


def _conv_geometry(conv):
    return (
        tuple(int(s) for s in conv.stride),
        tuple(int(p) for p in conv.padding),
        tuple(int(d) for d in conv.dilation),
        int(conv.groups),
    )


def _single_tensor_arg(args, kwargs, names):
    if len(args) == 1 and not kwargs and torch.is_tensor(args[0]):
        return args[0]
    if not args and len(kwargs) == 1:
        (key, value), = kwargs.items()
        if key in names and torch.is_tensor(value):
            return value
    return None


def _supported_input(hidden, device):
    return (
        torch.is_tensor(hidden)
        and hidden.dim() == 4
        and hidden.dtype is torch.float32
        and hidden.is_cuda
        and hidden.device == device
    )


def _probe_conv_relu(device):
    """Installation check that aten.cudnn_convolution_relu services this device
    for plain FP32 NCHW convolutions with the relu(conv+bias) semantics.
    Deterministic tensors only; synchronized so failures surface here."""
    try:
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16)
            w = torch.linspace(-0.05, 0.05, 9, device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
            b = torch.linspace(-0.02, 0.02, 1, device=device, dtype=torch.float32)
            out = _CUDNN_CONV_RELU(x, w, b, [1, 1], [1, 1], [1, 1], 1)
            ref = torch.relu(F.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), 1))
            torch.cuda.synchronize(device)
            return (
                tuple(out.shape) == tuple(ref.shape)
                and bool(torch.allclose(out, ref, atol=1e-5, rtol=1e-5))
                and bool((out >= 0).all())
            )
    except Exception:
        return False


def _probe_conv_add_relu(device):
    """Installation check that aten.cudnn_convolution_add_relu services this
    device for plain FP32 NCHW convolutions and computes
    relu(conv(x, W, bias) + 1.0 * z).  Deterministic tensors only."""
    try:
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16)
            w = torch.linspace(-0.05, 0.05, 9, device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
            b = torch.linspace(-0.02, 0.02, 1, device=device, dtype=torch.float32)
            z = torch.linspace(-0.5, 0.5, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16) * 0.5
            out = _CUDNN_CONV_ADD_RELU(x, w, z, 1.0, b, [1, 1], [1, 1], [1, 1], 1)
            ref = torch.relu(F.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), 1) + z)
            torch.cuda.synchronize(device)
            return (
                tuple(out.shape) == tuple(ref.shape)
                and bool(torch.allclose(out, ref, atol=1e-5, rtol=1e-5))
                and bool((out >= 0).all())
            )
    except Exception:
        return False


def _looks_like_basic_layer(module):
    layer = getattr(module, "layer", None)
    if not isinstance(layer, torch.nn.Sequential) or len(layer) != 2:
        return False
    for child in layer:
        if getattr(child, "convolution", None) is None or getattr(child, "normalization", None) is None:
            return False
    return (
        getattr(module, "shortcut", None) is not None
        and isinstance(getattr(module, "activation", None), torch.nn.ReLU)
    )


def _looks_like_shortcut(module):
    # ResNetShortCut: convolution + normalization only.  A ResNetConvLayer
    # additionally carries an ``activation`` (ReLU or Identity) and must NOT be
    # claimed here: this bundle owns its 17 conv sites through the embeddings
    # and block boundaries and declares no replacement for them.
    return (
        getattr(module, "convolution", None) is not None
        and getattr(module, "normalization", None) is not None
        and not hasattr(module, "activation")
        and not hasattr(module, "layer")
        and not hasattr(module, "embedder")
    )


def _looks_like_embeddings(module):
    embedder = getattr(module, "embedder", None)
    return (
        embedder is not None
        and getattr(embedder, "convolution", None) is not None
        and getattr(embedder, "normalization", None) is not None
        and isinstance(getattr(embedder, "activation", None), (torch.nn.ReLU, torch.nn.Identity))
        and getattr(module, "pooler", None) is not None
        and isinstance(getattr(module, "num_channels", None), int)
    )


def build(context):
    modules = context["modules"]
    references = context["references"]
    record_route = context.get("record_route", lambda *_: None)

    parameters = context.get("parameters") or {}
    epilogue_route = parameters.get("epilogue_route", "cudnn_conv_add_relu")
    if epilogue_route not in _EPILOGUE_ROUTES:
        epilogue_route = "cudnn_conv_add_relu"

    packed_cache = {}

    def packed_for(conv, bn):
        key = id(conv)
        if key not in packed_cache:
            packed_cache[key] = _pack_bn_into_conv(conv, bn)
        return packed_cache[key]

    replacements = {}

    for call_id, module in modules.items():
        parent = references[call_id]
        if _looks_like_embeddings(module):
            packed = packed_for(module.embedder.convolution, module.embedder.normalization)
            if packed is not None:
                replacements[call_id] = _make_embeddings_replacement(
                    call_id=call_id,
                    parent=parent,
                    module=module,
                    packed=packed,
                    record_route=record_route,
                )
        elif _looks_like_basic_layer(module):
            packed0 = packed_for(module.layer[0].convolution, module.layer[0].normalization)
            packed1 = packed_for(module.layer[1].convolution, module.layer[1].normalization)
            if packed0 is not None and packed1 is not None:
                replacements[call_id] = _make_block_replacement(
                    call_id=call_id,
                    parent=parent,
                    module=module,
                    packed0=packed0,
                    packed1=packed1,
                    requested_fused=epilogue_route == "cudnn_conv_add_relu",
                    record_route=record_route,
                )
        elif _looks_like_shortcut(module):
            packed = packed_for(module.convolution, module.normalization)
            if packed is not None:
                replacements[call_id] = _make_shortcut_replacement(
                    call_id=call_id,
                    parent=parent,
                    module=module,
                    packed=packed,
                    record_route=record_route,
                )

    # One installation probe per fused op on the (single) packed device, then
    # latch availability into every replacement.  Setup-only, tiny and
    # synchronized; failures route to the kernel-for-kernel parent sequences.
    devices = {packed[0].device for packed in packed_cache.values() if packed is not None}
    conv_relu_ok = False
    add_relu_ok = False
    if devices:
        device = next(iter(devices))
        conv_relu_ok = _probe_conv_relu(device)
        add_relu_ok = _probe_conv_add_relu(device)
    for replacement in replacements.values():
        replacement._opop_configure(conv_relu_ok, add_relu_ok)
    return replacements


def _make_embeddings_replacement(call_id, parent, module, packed, record_route):
    packed_weight, packed_bias = packed
    stride, padding, dilation, groups = _conv_geometry(module.embedder.convolution)
    device = packed_weight.device
    pooler = module.pooler
    embedder_modules = (module, module.embedder, module.embedder.convolution, module.embedder.normalization)
    num_channels = int(module.num_channels)

    state = {"conv_relu_ok": True, "setup_prefix": ""}

    def replacement(*args, **kwargs):
        pixel_values = _single_tensor_arg(args, kwargs, ("pixel_values", "hidden_states"))
        if pixel_values is None:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if not _supported_input(pixel_values, device) or int(pixel_values.shape[1]) != num_channels:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if any(m.training for m in embedder_modules):
            record_route(call_id, "parent:training_mode")
            return parent(*args, **kwargs)
        if torch.is_grad_enabled() and pixel_values.requires_grad:
            record_route(call_id, "parent:grad_context")
            return parent(*args, **kwargs)

        route = "aten.cudnn_convolution_relu"
        try:
            if state["conv_relu_ok"]:
                embedding = _CUDNN_CONV_RELU(
                    pixel_values, packed_weight, packed_bias,
                    list(stride), list(padding), list(dilation), groups,
                )
            else:
                route = state["setup_prefix"] + "conv2d_folded+relu"
                embedding = torch.relu(
                    F.conv2d(pixel_values, packed_weight, packed_bias, stride, padding, dilation, groups)
                )
        except Exception:
            state["conv_relu_ok"] = False
            route = "conv_relu_runtime_error->conv2d_folded+relu"
            embedding = torch.relu(
                F.conv2d(pixel_values, packed_weight, packed_bias, stride, padding, dilation, groups)
            )
        out = pooler(embedding)
        record_route(call_id, route)
        return out

    def _opop_configure(conv_relu_ok, add_relu_ok):
        state["conv_relu_ok"] = conv_relu_ok
        if not conv_relu_ok:
            state["setup_prefix"] = "setup:conv_relu_unavailable->"

    replacement._opop_configure = _opop_configure
    return replacement


def _make_block_replacement(call_id, parent, module, packed0, packed1, requested_fused, record_route):
    packed0_weight, packed0_bias = packed0
    packed1_weight, packed1_bias = packed1
    stride0, padding0, dilation0, groups0 = _conv_geometry(module.layer[0].convolution)
    stride1, padding1, dilation1, groups1 = _conv_geometry(module.layer[1].convolution)
    device = packed0_weight.device
    shortcut = module.shortcut
    block_modules = (
        module,
        module.layer[0], module.layer[1],
        module.layer[0].convolution, module.layer[0].normalization,
        module.layer[1].convolution, module.layer[1].normalization,
    )

    state = {
        "conv_relu_ok": True,
        "use_fused": bool(requested_fused),
        "fused_failed": False,
        "setup_prefix": "",
    }

    def replacement(*args, **kwargs):
        hidden_state = _single_tensor_arg(args, kwargs, ("hidden_state", "hidden_states"))
        if hidden_state is None:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if not _supported_input(hidden_state, device):
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if any(m.training for m in block_modules):
            record_route(call_id, "parent:training_mode")
            return parent(*args, **kwargs)
        if torch.is_grad_enabled() and hidden_state.requires_grad:
            record_route(call_id, "parent:grad_context")
            return parent(*args, **kwargs)

        # layer[0]: accepted folded conv+relu single kernel, with the
        # kernel-for-kernel fallback sequence on runtime decline.
        try:
            if state["conv_relu_ok"]:
                hidden = _CUDNN_CONV_RELU(
                    hidden_state, packed0_weight, packed0_bias,
                    list(stride0), list(padding0), list(dilation0), groups0,
                )
            else:
                hidden = torch.relu(
                    F.conv2d(hidden_state, packed0_weight, packed0_bias, stride0, padding0, dilation0, groups0)
                )
        except Exception:
            state["conv_relu_ok"] = False
            record_route(call_id, "conv_relu_runtime_error->conv2d_folded")
            hidden = torch.relu(
                F.conv2d(hidden_state, packed0_weight, packed0_bias, stride0, padding0, dilation0, groups0)
            )

        # Residual through the module boundary, exactly like the parent call:
        # nn.Identity keeps its parent module call, ResNetShortCut sites run
        # this bundle's installed folded replacement.
        residual = shortcut(hidden_state)

        # layer[1] + residual add + final ReLU.
        if state["use_fused"] and not state["fused_failed"]:
            try:
                out = _CUDNN_CONV_ADD_RELU(
                    hidden, packed1_weight, residual, 1.0, packed1_bias,
                    list(stride1), list(padding1), list(dilation1), groups1,
                )
                record_route(call_id, state["setup_prefix"] + "aten.cudnn_convolution_add_relu")
                return out
            except Exception:
                state["fused_failed"] = True
                record_route(call_id, "add_relu_runtime_error->conv2d_folded+add_+relu_")
        out = F.conv2d(hidden, packed1_weight, packed1_bias, stride1, padding1, dilation1, groups1)
        out += residual
        out = torch.relu(out)
        record_route(call_id, state["setup_prefix"] + "conv2d_folded+add_+relu_")
        return out

    def _opop_configure(conv_relu_ok, add_relu_ok):
        state["conv_relu_ok"] = conv_relu_ok
        if state["use_fused"] and not add_relu_ok:
            state["use_fused"] = False
            state["setup_prefix"] = "setup:add_relu_unavailable->"

    replacement._opop_configure = _opop_configure
    return replacement


def _make_shortcut_replacement(call_id, parent, module, packed, record_route):
    packed_weight, packed_bias = packed
    stride, padding, dilation, groups = _conv_geometry(module.convolution)
    device = packed_weight.device
    shortcut_modules = (module, module.convolution, module.normalization)

    state = {"setup_prefix": ""}

    def replacement(*args, **kwargs):
        input_tensor = _single_tensor_arg(args, kwargs, ("input", "hidden_state", "hidden_states"))
        if input_tensor is None:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if not _supported_input(input_tensor, device):
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if any(m.training for m in shortcut_modules):
            record_route(call_id, "parent:training_mode")
            return parent(*args, **kwargs)
        if torch.is_grad_enabled() and input_tensor.requires_grad:
            record_route(call_id, "parent:grad_context")
            return parent(*args, **kwargs)

        try:
            out = F.conv2d(input_tensor, packed_weight, packed_bias, stride, padding, dilation, groups)
        except Exception:
            record_route(call_id, state["setup_prefix"] + "conv2d_runtime_error->parent")
            return parent(*args, **kwargs)
        record_route(call_id, state["setup_prefix"] + "conv2d_folded")
        return out

    def _opop_configure(conv_relu_ok, add_relu_ok):
        pass

    replacement._opop_configure = _opop_configure
    return replacement
