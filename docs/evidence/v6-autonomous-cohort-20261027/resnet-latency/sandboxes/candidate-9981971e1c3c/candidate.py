"""Region candidate for the ResNet-18 offline batch-32 latency task (revision a345441f).

Explicitly replaces the whole accepted bundle ``trial-1bfb948628-param-aba0ae1b``
(12 module sites: eval-BN fold + fused ``aten.cudnn_convolution_relu`` /
``aten.cudnn_convolution_add_relu`` epilogues) with a single region replacement at
the ``ResNetForImageClassification.forward`` boundary (call
``f2cf62caf57ad09f5d2f36aa``, group ``eca3643289778aea5af3``).  The accepted
bundle's folded/fused conv semantics are re-issued unchanged inside the region;
the new mechanisms are:

  * whole-region CUDA-graph replay: the first guarded invocation runs the folded
    forward eagerly (warming every kernel, compiling Triton), the second guarded
    invocation captures the identical op sequence in the actual adapter
    invocation context (inside the installed replacement, never the original
    parent work) on a pre-warmed side stream and replays it, and every later
    invocation is one graph launch on the current input; when the incoming
    tensor's storage differs from the captured static input it is copied into
    the static buffer first, so the workload always executes on current inputs;
  * indices-free exact max pooling: the stem 3x3/stride-2/pad-1 max pool runs
    as one stride-generic Triton kernel computing the same maxima without the
    ~25.5 MB int32 index tensor the aten kernel writes (max selection is exact,
    so selected values are bit-identical; nothing downstream reads indices);
  * optional ``activation_layout='nhwc'`` route (TPE-measured either/or, never
    assumed): packed weights are converted to channels_last and the region runs
    with channels_last activations, one input-conversion kernel inside the graph.

Per-geometry fallback ladder preserves parent semantics: unsupported signature /
labels / hidden-states / return_dict / training / grad-context -> parent forward;
runtime failure of a fused op -> kernel-for-kernel folded ``conv2d`` sequence
(latched per site); Triton pool failure -> ``F.max_pool2d``; graph capture
failure -> permanent eager folded route.  All fallback categories carry concise
``record_route`` labels (``setup:...->``, ``parent:...``, ``...->eager``).

Setup work at build: structure validation of the pinned ResNet-18, eval-BN fold
packing of all 20 conv+BN pairs in float64 to contiguous float32 (~45 MB
retained packed weights; original parameters untouched so every parent fallback
recomputes the exact original computation), optional channels_last weight
conversion, and three tiny synchronized deterministic probes (no RNG) for the
two fused cudnn ops and the Triton pool.  The CUDA graph is captured lazily on
the second real invocation and retains its private pool (intermediates and
library workspaces), so reserved memory grows while per-run ``peak_allocated``
is not improved; this is documented in proposal.json and is not claimed as a
physical-memory saving.  No outputs are cached: every invocation replays the
full kernel sequence on the current input and returns a freshly cloned logits
tensor wrapped in the parent's output dataclass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention as _Output
except Exception:  # pragma: no cover - worker always has transformers
    _Output = None

_CUDNN_CONV_RELU = torch.ops.aten.cudnn_convolution_relu
_CUDNN_CONV_ADD_RELU = torch.ops.aten.cudnn_convolution_add_relu
_LAYOUTS = ("nchw", "nhwc")

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _maxpool_3x3_s2_p1(
        X, Y,
        C, H, W, OH, OW,
        sxn, sxc, sxh, sxw,
        syn, syc, syh, syw,
        BH: tl.constexpr, BW: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        oh = tl.program_id(1) * BH + tl.arange(0, BH)
        ow = tl.program_id(2) * BW + tl.arange(0, BW)
        n = pid_nc // C
        c = pid_nc % C
        mh = oh < OH
        mw = ow < OW
        acc = tl.full((BH, BW), float("-inf"), dtype=tl.float32)
        base = n * sxn + c * sxc
        for di in tl.static_range(3):
            ih = oh * 2 + (di - 1)
            vmh = mh & (ih >= 0) & (ih < H)
            for dj in tl.static_range(3):
                iw = ow * 2 + (dj - 1)
                m = vmh & mw & (iw >= 0) & (iw < W)
                offs = base + ih[:, None] * sxh + iw[None, :] * sxw
                v = tl.load(X + offs, mask=m, other=float("-inf"))
                acc = tl.maximum(acc, v)
        out_offs = n * syn + c * syc + oh[:, None] * syh + ow[None, :] * syw
        tl.store(Y + out_offs, acc, mask=mh & mw)

    _TRITON = True
except Exception:  # pragma: no cover - worker ships triton 3.7.1
    _TRITON = False


def _pair(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value), int(value))


def _pack_bn_into_conv(conv, bn, channels_last):
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
    if weight is None or weight.dtype is not torch.float32:
        return None
    with torch.no_grad():
        eps = float(bn.eps)
        scale = bn.weight.double() / (bn.running_var.double() + eps).sqrt()
        packed_weight = (weight.double() * scale.reshape(-1, 1, 1, 1)).to(torch.float32)
        bias64 = bn.bias.double() - bn.running_mean.double() * scale
        if conv.bias is not None:
            bias64 = bias64 + conv.bias.double() * scale
        packed_bias = bias64.to(torch.float32).contiguous()
    if channels_last:
        packed_weight = packed_weight.contiguous(memory_format=torch.channels_last)
    else:
        packed_weight = packed_weight.contiguous()
    return packed_weight, packed_bias


def _conv_geometry(conv):
    return (
        _pair(conv.stride),
        _pair(conv.padding),
        _pair(conv.dilation),
        int(conv.groups),
    )


def _conv_layer_parts(layer, need_relu):
    if not hasattr(layer, "convolution") or not hasattr(layer, "normalization"):
        return None
    activation = getattr(layer, "activation", None)
    if need_relu:
        if not isinstance(activation, torch.nn.ReLU):
            return None
    elif not isinstance(activation, torch.nn.Identity):
        return None
    return layer.convolution, layer.normalization


def _build_plan(module, channels_last):
    resnet = getattr(module, "resnet", None)
    encoder = getattr(resnet, "encoder", None)
    classifier = getattr(module, "classifier", None)
    embeddings = getattr(resnet, "embedder", None)
    head_pooler = getattr(resnet, "pooler", None)
    if (
        encoder is None
        or classifier is None
        or embeddings is None
        or not isinstance(head_pooler, torch.nn.AdaptiveAvgPool2d)
        or _pair(head_pooler.output_size) != (1, 1)
        or not isinstance(classifier, torch.nn.Sequential)
        or len(classifier) != 2
        or not isinstance(classifier[0], torch.nn.Flatten)
        or not isinstance(classifier[1], torch.nn.Linear)
    ):
        return None
    stem_parts = _conv_layer_parts(getattr(embeddings, "embedder", None), True)
    if stem_parts is None:
        return None
    stem_conv, stem_bn = stem_parts
    pooler = getattr(embeddings, "pooler", None)
    if (
        not isinstance(pooler, torch.nn.MaxPool2d)
        or _pair(pooler.kernel_size) != (3, 3)
        or _pair(pooler.stride) != (2, 2)
        or _pair(pooler.padding) != (1, 1)
        or _pair(pooler.dilation) != (1, 1)
        or bool(pooler.ceil_mode)
        or bool(getattr(pooler, "return_indices", False))
    ):
        return None

    stem_packed = _pack_bn_into_conv(stem_conv, stem_bn, channels_last)
    if stem_packed is None:
        return None

    linear = classifier[1]
    linear_weight = linear.weight
    if linear_weight is None or linear_weight.dtype is not torch.float32:
        return None
    linear_bias = linear.bias if isinstance(linear.bias, torch.Tensor) else None

    blocks = []
    devices = {stem_packed[0].device, linear_weight.device}
    stages = getattr(encoder, "stages", None)
    if not isinstance(stages, torch.nn.ModuleList):
        return None
    for stage in stages:
        layers = getattr(stage, "layers", None)
        if not isinstance(layers, torch.nn.Sequential):
            return None
        for block in layers:
            inner = getattr(block, "layer", None)
            if not isinstance(inner, torch.nn.Sequential) or len(inner) != 2:
                return None
            if not isinstance(getattr(block, "activation", None), torch.nn.ReLU):
                return None
            parts0 = _conv_layer_parts(inner[0], True)
            parts1 = _conv_layer_parts(inner[1], False)
            if parts0 is None or parts1 is None:
                return None
            packed0 = _pack_bn_into_conv(parts0[0], parts0[1], channels_last)
            packed1 = _pack_bn_into_conv(parts1[0], parts1[1], channels_last)
            if packed0 is None or packed1 is None:
                return None
            shortcut = getattr(block, "shortcut", None)
            if shortcut is None or isinstance(shortcut, torch.nn.Identity):
                shortcut_site = None
            else:
                sc_conv = getattr(shortcut, "convolution", None)
                sc_bn = getattr(shortcut, "normalization", None)
                if hasattr(shortcut, "activation") or sc_conv is None:
                    return None
                packed_s = _pack_bn_into_conv(sc_conv, sc_bn, channels_last)
                if packed_s is None:
                    return None
                shortcut_site = (packed_s, _conv_geometry(sc_conv))
                devices.add(packed_s[0].device)
            devices.add(packed0[0].device)
            devices.add(packed1[0].device)
            blocks.append(
                (
                    (packed0, _conv_geometry(parts0[0])),
                    shortcut_site,
                    (packed1, _conv_geometry(parts1[0])),
                )
            )
    if not blocks:
        return None
    if len(devices) != 1:
        return None
    return {
        "stem": (stem_packed, _conv_geometry(stem_conv)),
        "blocks": tuple(blocks),
        "num_channels": int(getattr(embeddings, "num_channels", stem_conv.in_channels)),
        "linear_weight": linear_weight.detach().clone(),
        "linear_bias": None if linear_bias is None else linear_bias.detach().clone(),
        "device": next(iter(devices)),
    }


def _supported_input(tensor, device):
    return (
        torch.is_tensor(tensor)
        and tensor.dim() == 4
        and tensor.dtype is torch.float32
        and tensor.device == device
    )


def _single_tensor_arg(args, kwargs):
    if len(args) == 1 and not kwargs and torch.is_tensor(args[0]):
        return args[0]
    if not args and kwargs:
        tensor = None
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                if tensor is not None or key != "pixel_values":
                    return None
                tensor = value
            elif value is not None:
                return None
        return tensor
    return None


def _geo(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _probe_conv_relu(device, channels_last):
    if device.type != "cuda":
        return False
    try:
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16)
            w = torch.linspace(-0.05, 0.05, 9, device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
            b = torch.linspace(-0.02, 0.02, 1, device=device, dtype=torch.float32)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
                w = w.contiguous(memory_format=torch.channels_last)
            out = _CUDNN_CONV_RELU(x, w, b, [1, 1], [1, 1], [1, 1], 1)
            ref = torch.relu(F.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), 1))
            torch.cuda.synchronize(device)
            return (
                tuple(out.shape) == tuple(ref.shape)
                and bool(torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-5))
                and bool((out >= 0).all())
            )
    except Exception:
        return False


def _probe_conv_add_relu(device, channels_last):
    if device.type != "cuda":
        return False
    try:
        with torch.no_grad():
            x = torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16)
            w = torch.linspace(-0.05, 0.05, 9, device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
            b = torch.linspace(-0.02, 0.02, 1, device=device, dtype=torch.float32)
            z = torch.linspace(-0.5, 0.5, 256, device=device, dtype=torch.float32).reshape(1, 1, 16, 16) * 0.5
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
                w = w.contiguous(memory_format=torch.channels_last)
                z = z.contiguous(memory_format=torch.channels_last)
            out = _CUDNN_CONV_ADD_RELU(x, w, z, 1.0, b, [1, 1], [1, 1], [1, 1], 1)
            ref = torch.relu(F.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), 1) + z)
            torch.cuda.synchronize(device)
            return (
                tuple(out.shape) == tuple(ref.shape)
                and bool(torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-5))
                and bool((out >= 0).all())
            )
    except Exception:
        return False


def _triton_maxpool(x):
    n, c, h, w = (int(v) for v in x.shape)
    oh = (h - 1) // 2 + 1
    ow = (w - 1) // 2 + 1
    if x.is_contiguous(memory_format=torch.channels_last):
        out = torch.empty(
            (n, c, oh, ow), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
        )
    else:
        out = torch.empty((n, c, oh, ow), device=x.device, dtype=x.dtype)
    sxn, sxc, sxh, sxw = x.stride()
    syn, syc, syh, syw = out.stride()
    grid = (n * c, triton.cdiv(oh, 32), triton.cdiv(ow, 32))
    _maxpool_3x3_s2_p1[grid](
        x, out, c, h, w, oh, ow,
        sxn, sxc, sxh, sxw, syn, syc, syh, syw,
        BH=32, BW=32,
    )
    return out


def _probe_triton_pool(device, channels_last):
    if device.type != "cuda" or not _TRITON:
        return False
    try:
        with torch.no_grad():
            x = torch.linspace(-2.0, 2.0, 600, device=device, dtype=torch.float32).reshape(2, 3, 10, 10)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            else:
                x = x.contiguous()
            ref = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            out = _triton_maxpool(x)
            torch.cuda.synchronize(device)
            return out.shape == ref.shape and bool(torch.equal(out, ref))
    except Exception:
        return False


def _make_region_replacement(call_id, module, parent, plan, layout, record_route):
    device = plan["device"]
    channels_last = layout == "nhwc"
    stem = plan["stem"]
    blocks = plan["blocks"]
    num_channels = plan["num_channels"]
    linear_weight = plan["linear_weight"]
    linear_bias = plan["linear_bias"]
    watched_modules = tuple(module.modules())

    conv_relu_avail = _probe_conv_relu(device, channels_last)
    add_relu_avail = _probe_conv_add_relu(device, channels_last)
    pool_triton_avail = _probe_triton_pool(device, channels_last)

    setup_notes = []
    if not conv_relu_avail:
        setup_notes.append("conv_relu_unavailable")
    if not add_relu_avail:
        setup_notes.append("add_relu_unavailable")
    if not pool_triton_avail:
        setup_notes.append("triton_pool_unavailable")
    prefix = ("setup:" + "+".join(setup_notes) + "->") if setup_notes else ""

    conv_relu_failed = set()
    add_relu_failed = set()
    pool_failed = False

    def to_layout(x):
        if channels_last:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def conv_relu_site(index, x, packed, geometry):
        weight, bias = packed
        stride, padding, dilation, groups = geometry
        if conv_relu_avail and index not in conv_relu_failed:
            try:
                return _CUDNN_CONV_RELU(
                    x, weight, bias, list(stride), list(padding), list(dilation), groups
                )
            except Exception:
                conv_relu_failed.add(index)
        return torch.relu(F.conv2d(x, weight, bias, stride, padding, dilation, groups))

    def epilogue_site(index, x, packed, residual, geometry):
        weight, bias = packed
        stride, padding, dilation, groups = geometry
        if add_relu_avail and index not in add_relu_failed:
            try:
                return _CUDNN_CONV_ADD_RELU(
                    x, weight, residual, 1.0, bias,
                    list(stride), list(padding), list(dilation), groups,
                )
            except Exception:
                add_relu_failed.add(index)
        out = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        out = out + residual
        return torch.relu(out)

    def shortcut_site(x, site):
        weight, bias = site[0]
        stride, padding, dilation, groups = site[1]
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)

    def pool(x):
        nonlocal pool_failed
        if pool_triton_avail and not pool_failed:
            try:
                return _triton_maxpool(x)
            except Exception:
                pool_failed = True
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    def eager_forward(x):
        hidden = conv_relu_site(-1, to_layout(x), stem[0], stem[1])
        hidden = pool(hidden)
        for index, (site0, shortcut, site1) in enumerate(blocks):
            residual = shortcut_site(hidden, shortcut) if shortcut is not None else hidden
            hidden = conv_relu_site(index, hidden, site0[0], site0[1])
            hidden = epilogue_site(index, hidden, site1[0], residual, site1[1])
        pooled = F.adaptive_avg_pool2d(hidden, (1, 1)).flatten(1)
        if linear_bias is None:
            return F.linear(pooled, linear_weight)
        return F.linear(pooled, linear_weight, linear_bias)

    graph_state = {
        "calls": 0,
        "graph": None,
        "static_in": None,
        "static_out": None,
        "geo": None,
        "failed": False,
    }

    def capture_into(x):
        torch.cuda.synchronize()
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager_forward(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            static_out = eager_forward(x)
        graph_state["graph"] = graph
        graph_state["static_in"] = x
        graph_state["static_out"] = static_out
        graph_state["geo"] = _geo(x)

    def wrap(logits):
        if _Output is not None:
            return _Output(loss=None, logits=logits, hidden_states=None)
        return type("Output", (), {"logits": logits})()

    def eager_or_parent(x, args, kwargs, label):
        try:
            logits = eager_forward(x)
        except Exception:
            record_route(call_id, label + "->parent")
            return parent(*args, **kwargs)
        record_route(call_id, label)
        return wrap(logits)

    def replacement(*args, **kwargs):
        pixel_values = _single_tensor_arg(args, kwargs)
        if pixel_values is None or not _supported_input(pixel_values, device):
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if int(pixel_values.shape[1]) != num_channels:
            record_route(call_id, "parent:unsupported_signature")
            return parent(*args, **kwargs)
        if any(m.training for m in watched_modules):
            record_route(call_id, "parent:training_mode")
            return parent(*args, **kwargs)
        if torch.is_grad_enabled() and pixel_values.requires_grad:
            record_route(call_id, "parent:grad_context")
            return parent(*args, **kwargs)

        graph_state["calls"] += 1
        graph = graph_state["graph"]

        if graph is not None:
            if _geo(pixel_values) == graph_state["geo"]:
                if pixel_values.data_ptr() != graph_state["static_in"].data_ptr():
                    graph_state["static_in"].copy_(pixel_values, non_blocking=True)
                graph.replay()
                record_route(call_id, prefix + layout + ":cuda_graph_replay")
                return wrap(graph_state["static_out"].clone())
            return eager_or_parent(
                pixel_values, args, kwargs, prefix + layout + ":eager_shape_mismatch"
            )

        if graph_state["failed"]:
            return eager_or_parent(
                pixel_values, args, kwargs, prefix + layout + ":eager_no_graph"
            )

        if graph_state["calls"] == 1 or pixel_values.device.type != "cuda":
            return eager_or_parent(
                pixel_values, args, kwargs, prefix + layout + ":eager_warmup"
            )

        try:
            capture_into(pixel_values)
        except Exception:
            graph_state["failed"] = True
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            return eager_or_parent(
                pixel_values, args, kwargs, prefix + layout + ":graph_capture_failed"
            )
        graph_state["graph"].replay()
        record_route(call_id, prefix + layout + ":cuda_graph_capture+replay")
        return wrap(graph_state["static_out"].clone())

    return replacement


def build(context):
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route", lambda *_: None)
    parameters = context.get("parameters") or {}
    layout = parameters.get("activation_layout", "nchw")
    if layout not in _LAYOUTS:
        layout = "nchw"

    replacements = {}
    for call_id, module in modules.items():
        if call_id not in references:
            continue
        plan = _build_plan(module, channels_last=(layout == "nhwc"))
        if plan is None:
            continue
        replacements[call_id] = _make_region_replacement(
            call_id, module, references[call_id], plan, layout, record_route
        )
    return replacements
