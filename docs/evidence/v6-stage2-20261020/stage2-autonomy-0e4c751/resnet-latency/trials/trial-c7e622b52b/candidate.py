"""Composite candidate for the ResNet pooler + residual-block region.

Explicitly replaces the accepted conv-fold bundle. The 20 conv sites keep
folded-BN execution (cudnn epilogue when calibrated), the MaxPool2d pooler
runs an index-free Triton 3x3/s2/p1 max kernel, and each ResNetBasicLayer
dispatches through its layer[0]/layer[1]/shortcut module boundaries (so the
installed conv-site replacements execute and are counted) while fusing the
residual epilogue: preferred tier routes layer.1 through
aten::cudnn_convolution_add_relu (conv+bias+add+relu in one epilogue) via a
verified residual handoff, otherwise one Triton add+relu kernel replaces the
separate add_ and relu passes. Every tier is calibrated at installation
against the raw parent on synthetic inputs and falls back to the parent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ATOL = 1e-4
_RTOL = 1e-4

try:
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:
    _TRITON = False

if _TRITON:

    @triton.jit
    def _nan_max(a, b):
        an = a != a
        bn = b != b
        return tl.where(an, a, tl.where(bn, b, tl.maximum(a, b)))

    @triton.jit
    def _maxpool3x3_s2(X, Y, H, W, OH, OW, BLOCK_W: tl.constexpr):
        pid = tl.program_id(0)
        nc = pid // OH
        oh = pid - nc * OH
        ow = tl.arange(0, BLOCK_W)
        ow_m = ow < OW
        base = X + (nc * H * W).to(tl.int64)
        h0 = oh * 2 - 1
        a0 = tl.full((BLOCK_W,), float("-inf"), dtype=tl.float32)
        a1 = tl.full((BLOCK_W,), float("-inf"), dtype=tl.float32)
        a2 = tl.full((BLOCK_W,), float("-inf"), dtype=tl.float32)
        for k in tl.static_range(3):
            ww = ow * 2 + (k - 1)
            cm = (ww >= 0) & (ww < W) & ow_m
            hh = h0
            m = cm & (hh >= 0) & (hh < H)
            v = tl.load(base + hh * W + ww, mask=m, other=float("-inf"))
            a0 = _nan_max(a0, v)
            hh = h0 + 1
            m = cm & (hh >= 0) & (hh < H)
            v = tl.load(base + hh * W + ww, mask=m, other=float("-inf"))
            a1 = _nan_max(a1, v)
            hh = h0 + 2
            m = cm & (hh >= 0) & (hh < H)
            v = tl.load(base + hh * W + ww, mask=m, other=float("-inf"))
            a2 = _nan_max(a2, v)
        out = _nan_max(_nan_max(a0, a1), a2)
        tl.store(Y + (pid * OW).to(tl.int64) + ow, out, mask=ow_m)

    @triton.jit
    def _add_relu(A, B, OUT, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        a = tl.load(A + offs, mask=m)
        b = tl.load(B + offs, mask=m)
        s = a + b
        r = tl.where(s != s, s, tl.maximum(s, 0.0))
        tl.store(OUT + offs, r, mask=m)


def _op(name):
    try:
        op = getattr(torch.ops.aten, name)
        _ = op.default._schema
        return op.default
    except Exception:
        return None


_CONV_RELU = _op("cudnn_convolution_relu")
_CONV_ADD_RELU = _op("cudnn_convolution_add_relu")


def _triton_pool(x):
    N, C, H, W = (int(v) for v in x.shape)
    if H < 2 or W < 2 or x.numel() >= 2**31:
        return None
    OH = (H - 1) // 2 + 1
    OW = (W - 1) // 2 + 1
    block_w = int(triton.next_power_of_2(OW))
    if block_w > 256:
        return None
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    _maxpool3x3_s2[(N * C * OH,)](x, out, H, W, OH, OW, BLOCK_W=block_w, num_warps=4)
    return out


def _fused_add_relu_(a, b):
    if (
        a is None
        or b is None
        or a.shape != b.shape
        or a.dtype != torch.float32
        or b.dtype != torch.float32
        or not a.is_cuda
    ):
        return torch.relu_(a.add_(b))
    n = a.numel()
    if _TRITON and n > 0 and n < 2**31 and a.is_contiguous() and b.is_contiguous():
        try:
            _add_relu[(triton.cdiv(n, 1024),)](a, b, a, n, BLOCK=1024, num_warps=4)
            return a
        except Exception:
            pass
    return torch.relu_(a.add_(b))


def _fold(conv, bn):
    weight = conv.weight
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    if weight is None or gamma is None or beta is None or running_mean is None or running_var is None:
        return None
    with torch.no_grad():
        scale = (gamma / torch.sqrt(running_var + bn.eps)).to(dtype=weight.dtype)
        folded_w = (weight * scale.view(-1, 1, 1, 1)).contiguous()
        folded_b = beta - running_mean * scale
        if conv.bias is not None:
            folded_b = folded_b + conv.bias * scale
        folded_b = folded_b.contiguous()
    return folded_w, folded_b


def _close(actual, ref):
    try:
        if actual.shape != ref.shape or actual.dtype != ref.dtype or actual.device != ref.device:
            return False
        diff = (actual - ref).abs()
        tol = _ATOL + _RTOL * ref.abs()
        return bool(torch.all(diff <= tol).item())
    except Exception:
        return False


def _time_ms(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _observed_shape(spec, rank):
    for signatures in (
        (spec.get("rank_specs") or {}).get(str(rank), {}).get("signatures"),
        spec.get("signatures"),
    ):
        if not signatures:
            continue
        args = signatures[0].get("args") or []
        if args and isinstance(args[0], dict) and args[0].get("shape"):
            shape = [int(v) for v in args[0]["shape"]]
            if len(shape) == 4:
                return shape
    return None


def _synthetic(shape, call_id):
    seed = 0x51ED2705 + (sum(ord(c) for c in call_id) % 1000003)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return x.cuda()


def _site_input(shape, in_ch, kernel):
    if shape is None or shape[1] != in_ch:
        shape = [2, in_ch, 8 * kernel + 8, 8 * kernel + 8]
    return shape


def _calibrate_conv(call_id, conv, bn, parent, fuse_relu, shape):
    if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
        return "parent", "setup_unsupported_normalization"
    if not getattr(bn, "track_running_stats", False):
        return "parent", "setup_fold_unavailable"
    if fuse_relu is None:
        return "parent", "setup_unsupported_activation"
    device = conv.weight.device
    stride = [int(v) for v in conv.stride]
    padding = [int(v) for v in conv.padding]
    dilation = [int(v) for v in conv.dilation]
    groups = int(conv.groups)
    in_ch = int(conv.in_channels)
    kernel = max(int(conv.kernel_size[0]), 1)
    if shape is None or shape[1] != in_ch:
        shape = [2, in_ch, 8 * kernel + 8, 8 * kernel + 8]
    x = _synthetic(shape, call_id)
    try:
        with torch.no_grad():
            folded = _fold(conv, bn)
            if folded is None:
                return "parent", "setup_fold_unavailable"
            w, b = folded
            ref = parent(x)
            if not isinstance(ref, torch.Tensor):
                return "parent", "setup_parent_signature"

            def run_conv2d():
                out = F.conv2d(x, w, b, stride, padding, dilation, groups)
                if fuse_relu:
                    out = torch.relu_(out)
                return out

            candidates = {"conv2d_folded": run_conv2d}
            if fuse_relu and _CONV_RELU is not None:

                def run_fused():
                    return _CONV_RELU(x, w, b, stride, padding, dilation, groups)

                candidates["cudnn_fused"] = run_fused

            compliant = {}
            for name, fn in candidates.items():
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    if _close(y, ref):
                        compliant[name] = fn
                    del y
                except Exception:
                    continue
            if not compliant:
                return "parent", "setup_calibration_failed"
            parent_ms = None
            try:
                parent_ms = _time_ms(lambda: parent(x))
            except Exception:
                parent_ms = None
            times = {name: _time_ms(fn) for name, fn in compliant.items()}
            best = min(times, key=times.get)
            if "cudnn_fused" in times and times["cudnn_fused"] <= times[best] * 1.02:
                best = "cudnn_fused"
            if parent_ms is not None and parent_ms < times[best] * 0.90:
                return "parent", "setup_parent_fastest"
            return best, "calibrated:" + best
    finally:
        del x
        torch.cuda.synchronize()


def _make_conv(call_id, module, parent, record_route, tier, fuse_relu, shared=None, alpha_holder=None):
    conv = getattr(module, "convolution", None)
    bn = getattr(module, "normalization", None)
    if not isinstance(conv, nn.Conv2d) or bn is None:
        tier, fuse_relu = "parent", False
        stride = padding = dilation = []
        groups = 1
        in_ch = -1
        device = None
    else:
        stride = [int(v) for v in conv.stride]
        padding = [int(v) for v in conv.padding]
        dilation = [int(v) for v in conv.dilation]
        groups = int(conv.groups)
        in_ch = int(conv.in_channels)
        device = conv.weight.device
    cache = {"key": None, "w": None, "b": None}

    def _ensure_fold():
        bias = conv.bias
        key = (
            conv.weight._version,
            bn.weight._version,
            bn.bias._version,
            bn.running_mean._version,
            bn.running_var._version,
            -1 if bias is None else bias._version,
        )
        if cache["key"] != key:
            folded = _fold(conv, bn)
            cache["w"], cache["b"] = folded if folded is not None else (None, None)
            cache["key"] = key
        return cache["w"], cache["b"]

    def replacement(*args, **kwargs):
        if tier == "parent":
            record_route(call_id, "setup_route_parent")
            return parent(*args, **kwargs)
        x = None
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
        else:
            for name in ("hidden_states", "input"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor):
                    x = value
                    break
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 4
            or x.dtype != torch.float32
            or x.device != device
            or int(x.shape[1]) != in_ch
            or module.training
            or conv.training
            or bn.training
            or torch.is_grad_enabled()
        ):
            record_route(call_id, "guard_unsupported_signature_parent")
            return parent(*args, **kwargs)
        w, b = _ensure_fold()
        if w is None or b is None:
            record_route(call_id, "guard_fold_unavailable_parent")
            return parent(*args, **kwargs)
        if tier == "cudnn_add_relu":
            res = shared.get("residual") if shared is not None else None
            if (
                isinstance(res, torch.Tensor)
                and res.dim() == 4
                and list(res.shape) == list(x.shape)
                and res.dtype == torch.float32
                and res.device == w.device
            ):
                try:
                    alpha = alpha_holder[0] if alpha_holder is not None else None
                    out = _CONV_ADD_RELU(x, w, res, alpha, b, stride, padding, dilation, groups)
                    shared["fused"] = True
                    record_route(call_id, "cudnn_conv_bias_add_relu_fused")
                    return out
                except Exception:
                    pass
        try:
            if tier == "cudnn_fused":
                out = _CONV_RELU(x, w, b, stride, padding, dilation, groups)
                record_route(call_id, "cudnn_conv_bias_relu_fused")
                return out
            out = F.conv2d(x, w, b, stride, padding, dilation, groups)
            if fuse_relu:
                torch.relu_(out)
            record_route(call_id, "conv2d_bn_folded_relu" if fuse_relu else "conv2d_bn_folded")
            return out
        except Exception:
            record_route(call_id, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def _norm2(v):
    if isinstance(v, (tuple, list)):
        return tuple(int(i) for i in v)
    return (int(v), int(v))


def _calibrate_pool(call_id, module, parent, spec, rank):
    if not _TRITON:
        return "parent", "setup_triton_unavailable"
    cfg_ok = (
        _norm2(module.kernel_size) == (3, 3)
        and _norm2(module.stride) == (2, 2)
        and _norm2(module.padding) == (1, 1)
        and _norm2(module.dilation) == (1, 1)
        and not bool(module.ceil_mode)
        and not bool(module.return_indices)
    )
    if not cfg_ok:
        return "parent", "setup_unsupported_module"
    shape = _observed_shape(spec, rank) or [32, 64, 112, 112]
    x = _synthetic(shape, call_id)
    try:
        with torch.no_grad():
            ref = parent(x)
            try:
                out = _triton_pool(x)
                torch.cuda.synchronize()
            except Exception:
                return "parent", "setup_triton_unavailable"
            if out is None or not torch.equal(out, ref):
                return "parent", "setup_calibration_failed"
            parent_ms = None
            try:
                parent_ms = _time_ms(lambda: parent(x))
            except Exception:
                parent_ms = None
            triton_ms = _time_ms(lambda: _triton_pool(x))
            if parent_ms is not None and parent_ms < triton_ms * 0.90:
                return "parent", "setup_parent_fastest"
            return "triton", "calibrated:triton_maxpool3x3s2"
    finally:
        del x
        torch.cuda.synchronize()


def _make_pool(call_id, module, parent, record_route, tier, setup_label):
    cfg_ok = (
        _norm2(module.kernel_size) == (3, 3)
        and _norm2(module.stride) == (2, 2)
        and _norm2(module.padding) == (1, 1)
        and _norm2(module.dilation) == (1, 1)
        and not bool(module.ceil_mode)
        and not bool(module.return_indices)
    )

    def replacement(*args, **kwargs):
        if tier != "triton" or not cfg_ok:
            record_route(call_id, "setup_route_parent:" + setup_label if tier != "triton" else "setup_route_parent:setup_unsupported_module")
            return parent(*args, **kwargs)
        x = None
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
        else:
            value = kwargs.get("input")
            if isinstance(value, torch.Tensor):
                x = value
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 4
            or x.dtype != torch.float32
            or not x.is_cuda
            or not x.is_contiguous()
            or module.training
            or torch.is_grad_enabled()
        ):
            record_route(call_id, "guard_unsupported_signature_parent")
            return parent(*args, **kwargs)
        try:
            out = _triton_pool(x)
            if out is None:
                record_route(call_id, "guard_unsupported_signature_parent")
                return parent(*args, **kwargs)
            record_route(call_id, "triton_maxpool3x3s2")
            return out
        except Exception:
            record_route(call_id, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def _calibrate_basic(call_id, module, parent, spec, rank, l0_repl, l1_plain, l1_fused, sc_repl, shared, alpha_holder):
    layer0 = module.layer[0]
    shortcut = module.shortcut
    identity_shortcut = isinstance(shortcut, nn.Identity)
    conv0 = getattr(layer0, "convolution", None)
    if not isinstance(conv0, nn.Conv2d):
        return "parent", "setup_unsupported_module"
    in_ch = int(conv0.in_channels)
    shape = _observed_shape(spec, rank)
    if shape is None or shape[1] != in_ch:
        shape = [2, in_ch, 64, 64]
    x = _synthetic(shape, call_id)
    try:
        with torch.no_grad():
            ref = parent(x)
            if not isinstance(ref, torch.Tensor):
                return "parent", "setup_parent_signature"

            def res_of():
                return x if identity_shortcut else sc_repl(x)

            def run_triton():
                l0 = l0_repl(x)
                l1 = l1_plain(l0)
                res = res_of()
                return _fused_add_relu_(l1, res)

            candidates = {"triton_add_relu": run_triton}

            def run_fused():
                l0 = l0_repl(x)
                res = res_of()
                shared["residual"] = res
                shared["fused"] = False
                try:
                    l1 = l1_fused(l0)
                finally:
                    shared.pop("residual", None)
                    fused = shared.pop("fused", False)
                return l1 if fused else _fused_add_relu_(l1, res)

            if _CONV_ADD_RELU is not None and l1_fused is not None:
                for alpha in (None, 1.0):
                    alpha_holder[0] = alpha
                    try:
                        y = run_fused()
                        torch.cuda.synchronize()
                        if _close(y, ref):
                            candidates["cudnn_fused_epilogue"] = run_fused
                            break
                    except Exception:
                        continue
                    finally:
                        if "cudnn_fused_epilogue" not in candidates:
                            alpha_holder[0] = None
                if "cudnn_fused_epilogue" not in candidates:
                    alpha_holder[0] = None

            compliant = {}
            for name, fn in candidates.items():
                try:
                    y = fn()
                    torch.cuda.synchronize()
                    if _close(y, ref):
                        compliant[name] = fn
                    del y
                except Exception:
                    continue
            if not compliant:
                return "parent", "setup_calibration_failed"
            parent_ms = None
            try:
                parent_ms = _time_ms(lambda: parent(x))
            except Exception:
                parent_ms = None
            times = {name: _time_ms(fn) for name, fn in compliant.items()}
            best = min(times, key=times.get)
            if "cudnn_fused_epilogue" in times and times["cudnn_fused_epilogue"] <= times[best] * 1.02:
                best = "cudnn_fused_epilogue"
            if parent_ms is not None and parent_ms < times[best] * 0.90:
                return "parent", "setup_parent_fastest"
            return best, "calibrated:" + best
    finally:
        del x
        torch.cuda.synchronize()


def _make_basic(call_id, module, parent, record_route, tier, shared):
    layer0 = getattr(module, "layer", None)
    shortcut = getattr(module, "shortcut", None)
    if layer0 is None or shortcut is None or len(layer0) < 2:
        tier = "parent"
        in_ch = -1
        device = None
    else:
        conv0 = getattr(layer0[0], "convolution", None)
        if not isinstance(conv0, nn.Conv2d):
            tier = "parent"
            in_ch = -1
            device = None
        else:
            in_ch = int(conv0.in_channels)
            device = conv0.weight.device
    identity_shortcut = isinstance(shortcut, nn.Identity)

    def replacement(*args, **kwargs):
        if tier == "parent":
            record_route(call_id, "setup_route_parent")
            return parent(*args, **kwargs)
        x = None
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
        else:
            for name in ("hidden_state", "hidden_states", "input"):
                value = kwargs.get(name)
                if isinstance(value, torch.Tensor):
                    x = value
                    break
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 4
            or x.dtype != torch.float32
            or x.device != device
            or int(x.shape[1]) != in_ch
            or module.training
            or torch.is_grad_enabled()
        ):
            record_route(call_id, "guard_unsupported_signature_parent")
            return parent(*args, **kwargs)
        try:
            l0 = module.layer[0](x)
            res = x if identity_shortcut else module.shortcut(x)
            if tier == "fused":
                shared["residual"] = res
                shared["fused"] = False
                try:
                    l1 = module.layer[1](l0)
                finally:
                    shared.pop("residual", None)
                    fused = shared.pop("fused", False)
                if fused:
                    record_route(call_id, "basic_layer_conv_bias_add_relu_fused")
                    return l1
            else:
                l1 = module.layer[1](l0)
            out = _fused_add_relu_(l1, res)
            record_route(call_id, "basic_layer_triton_add_relu")
            return out
        except Exception:
            record_route(call_id, "runtime_failure_parent")
            return parent(*args, **kwargs)

    return replacement


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    targets = context.get("targets") or {}
    rank = context.get("rank", 0)
    replacements = {}

    module_specs = {}
    for cid, spec in targets.items():
        try:
            if (spec or {}).get("kind") != "module" or cid not in modules:
                continue
            module_specs[cid] = spec
        except Exception:
            continue

    by_path = {}
    for cid, spec in module_specs.items():
        by_path[str(spec.get("target") or "")] = cid

    for cid, spec in module_specs.items():
        sem = str(spec.get("semantics") or "")
        if not (sem.endswith("ResNetConvLayer") or sem.endswith("ResNetShortCut")):
            continue
        module = modules[cid]
        parent = references.get(cid)
        if parent is None:
            continue
        conv = getattr(module, "convolution", None)
        bn = getattr(module, "normalization", None)
        act = getattr(module, "activation", None)
        if not isinstance(conv, nn.Conv2d) or bn is None:
            replacements[cid] = _make_conv(cid, module, parent, record_route, "parent", False)
            continue
        if isinstance(act, nn.ReLU):
            fuse_relu = True
        elif act is None or isinstance(act, nn.Identity):
            fuse_relu = False
        else:
            fuse_relu = None
        try:
            tier, label = _calibrate_conv(cid, conv, bn, parent, fuse_relu, _observed_shape(spec, rank))
        except Exception:
            tier, label = "parent", "setup_exception_parent"
        replacements[cid] = _make_conv(cid, module, parent, record_route, tier, fuse_relu)

    for cid, spec in module_specs.items():
        sem = str(spec.get("semantics") or "")
        if not sem.endswith("MaxPool2d"):
            continue
        module = modules[cid]
        parent = references.get(cid)
        if parent is None:
            continue
        try:
            tier, label = _calibrate_pool(cid, module, parent, spec, rank)
        except Exception:
            tier, label = "parent", "setup_exception_parent"
        replacements[cid] = _make_pool(cid, module, parent, record_route, tier, label)

    for cid, spec in module_specs.items():
        sem = str(spec.get("semantics") or "")
        if not sem.endswith("ResNetBasicLayer"):
            continue
        module = modules[cid]
        parent = references.get(cid)
        if parent is None:
            continue
        path = str(spec.get("target") or "")
        l0_cid = by_path.get(path + ".layer.0")
        l1_cid = by_path.get(path + ".layer.1")
        sc_cid = by_path.get(path + ".shortcut")
        l0_repl = replacements.get(l0_cid) if l0_cid is not None else None
        l1_plain = replacements.get(l1_cid) if l1_cid is not None else None
        sc_repl = replacements.get(sc_cid) if sc_cid is not None else None
        layer_seq = getattr(module, "layer", None)
        block_ok = (
            isinstance(layer_seq, nn.Sequential)
            and len(layer_seq) == 2
            and isinstance(getattr(module, "activation", None), nn.ReLU)
            and (
                getattr(layer_seq[1], "activation", None) is None
                or isinstance(getattr(layer_seq[1], "activation", None), nn.Identity)
            )
        )
        usable = (
            block_ok
            and l0_repl is not None
            and l1_plain is not None
            and (isinstance(getattr(module, "shortcut", None), nn.Identity) or sc_repl is not None)
        )
        shared = {}
        alpha_holder = [None]
        if not usable:
            replacements[cid] = _make_basic(cid, module, parent, record_route, "parent", {})
            continue
        l1_fused = None
        if _CONV_ADD_RELU is not None and l1_cid is not None:
            l1_act = getattr(modules[l1_cid], "activation", None)
            l1_fuse_relu = isinstance(l1_act, nn.ReLU)
            l1_fused = _make_conv(
                l1_cid,
                modules[l1_cid],
                references.get(l1_cid),
                record_route,
                "cudnn_add_relu",
                l1_fuse_relu,
                shared,
                alpha_holder,
            )
        try:
            tier, label = _calibrate_basic(
                cid,
                module,
                parent,
                spec,
                rank,
                l0_repl,
                l1_plain,
                l1_fused,
                sc_repl if sc_repl is not None else (lambda t: t),
                shared,
                alpha_holder,
            )
        except Exception:
            tier, label = "parent", "setup_exception_parent"
        if tier == "cudnn_fused_epilogue" and l1_fused is not None and l1_cid is not None:
            replacements[l1_cid] = l1_fused
        basic_tier = {"cudnn_fused_epilogue": "fused", "triton_add_relu": "triton"}.get(tier, "parent")
        replacements[cid] = _make_basic(cid, module, parent, record_route, basic_tier, shared)

    return replacements
