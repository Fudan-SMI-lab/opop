"""Whole-bundle fused FP32 inference replacement for the ResNetForImageClassification
module call f2cf62caf57ad09f5d2f36aa, composing on the installed bundle
trial-9c6f5280d7-param-ad514673 (which this candidate explicitly replaces).

Everything from the accepted bundle is preserved: eval BatchNorm2d folded into
bias-free conv weights, fused residual add+bias+ReLU and conv bias+ReLU Triton
pointwise kernels, identical aten conv2d/max_pool2d/adaptive_avg_pool2d/linear
ops under the frozen FP32/no-TF32 policy, lazy side-stream warmup -> CUDA-graph
capture -> verified replay with input-signature guards, eager fused path and
parent fallback for non-matching signatures.

New in this revision: the stem epilogue is fused. The accepted bundle first
writes the 32x64x112x112 stem activation with a separate _bias_relu kernel
(measured 210.0us) and then max_pool_forward_nchw re-reads that activation and
writes the 32x64x56x56 pooled tensor (measured 183.6us at ~1.26TB/s, the least
efficient kernel in the window). Per-channel bias and ReLU commute with a
spatial max for finite floats (IEEE addition and ReLU are monotonic, so
max_i relu(x_i + b) == relu(max_i x_i + b) bitwise), so one fused Triton kernel
now reads the raw conv output once, applies bias+ReLU while pooling, and writes
only the pooled tensor.

The stem pooler installed in transformers 5.16.1 is
nn.MaxPool2d(kernel_size=3, stride=2, padding=1) (ResNetEmbeddings.__init__),
so the fused kernel implements exactly that geometry with masked loads for the
padded border; the fusion guard accepts only k=(3,3), s=(2,2), p=(1,1),
d=(1,1), ceil_mode=False and every other geometry keeps the accepted bundle's
separate relu(x+b) then F.max_pool2d sequence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CALL_ID = "f2cf62caf57ad09f5d2f36aa"

try:
    from transformers.modeling_outputs import (
        ImageClassifierOutputWithNoAttention as _Output,
    )
except Exception:  # pragma: no cover
    _Output = None


def _pair(v):
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


def _conv_params(conv):
    return (
        _pair(conv.stride),
        _pair(conv.padding),
        _pair(conv.dilation),
        int(conv.groups),
    )


def _pool_params(pool):
    return (
        _pair(pool.kernel_size),
        _pair(pool.stride),
        _pair(pool.padding),
        _pair(pool.dilation),
        bool(pool.ceil_mode),
    )


def _fold(conv, bn):
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    if bn.training:
        raise ValueError("batch norm not in eval mode")
    if conv.weight.dim() != 4 or bn.num_features != conv.out_channels:
        raise ValueError("unsupported conv/bn geometry")
    inv = torch.rsqrt(var + bn.eps)
    scale = gamma * inv
    weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    shift = beta - mean * scale
    if conv.bias is not None:
        shift = shift + conv.bias.detach()
    return weight.contiguous(), shift.contiguous()


def _require(module, cls):
    if not isinstance(module, cls):
        raise ValueError(f"unexpected module {type(module)!r}, expected {cls.__name__}")


def _build_plan(module):
    import torch.nn as nn

    if _Output is None:
        raise ValueError("transformers output type unavailable")
    resnet = getattr(module, "resnet", None)
    _require(resnet, nn.Module)
    emb = resnet.embedder
    _require(emb.embedder, __import__("transformers.models.resnet.modeling_resnet", fromlist=["ResNetConvLayer"]).ResNetConvLayer)
    _require(emb.pooler, nn.MaxPool2d)
    _require(resnet.pooler, nn.AdaptiveAvgPool2d)
    stem = emb.embedder
    _require(stem.convolution, nn.Conv2d)
    _require(stem.normalization, nn.BatchNorm2d)
    _require(stem.activation, nn.ReLU)
    steps = []
    w, b = _fold(stem.convolution, stem.normalization)
    mp = emb.pooler
    k, s, p, d, ceil = _pool_params(mp)
    if k == (3, 3) and s == (2, 2) and p == (1, 1) and d == (1, 1) and not ceil:
        # Fusable stem epilogue (the geometry installed by transformers
        # ResNetEmbeddings: MaxPool2d(kernel_size=3, stride=2, padding=1)):
        # bias+ReLU and the spatial max run in one kernel because
        # relu(max(x)+b) == max(relu(x+b)) for finite floats.
        steps.append(
            {
                "kind": "stem_pool",
                "w": w,
                "b": b,
                "p": _conv_params(stem.convolution),
                "k": k,
                "s": s,
                "pp": p,
                "d": d,
                "ceil": ceil,
            }
        )
    else:
        steps.append({"kind": "stem", "w": w, "b": b, "p": _conv_params(stem.convolution)})
        steps.append({"kind": "maxpool", "k": k, "s": s, "p": p, "d": d, "ceil": ceil})
    modeling = __import__("transformers.models.resnet.modeling_resnet", fromlist=["ResNetBasicLayer", "ResNetShortCut"])
    for stage in resnet.encoder.stages:
        for layer in stage.layers:
            _require(layer, modeling.ResNetBasicLayer)
            c1, c2 = layer.layer[0], layer.layer[1]
            _require(c1, modeling.ResNetConvLayer)
            _require(c2, modeling.ResNetConvLayer)
            _require(c1.convolution, nn.Conv2d)
            _require(c1.normalization, nn.BatchNorm2d)
            _require(c1.activation, nn.ReLU)
            _require(c2.convolution, nn.Conv2d)
            _require(c2.normalization, nn.BatchNorm2d)
            if not isinstance(c2.activation, nn.Identity):
                raise ValueError("second conv activation is not identity")
            _require(layer.activation, nn.ReLU)
            w1, b1 = _fold(c1.convolution, c1.normalization)
            w2, b2 = _fold(c2.convolution, c2.normalization)
            sc = layer.shortcut
            if isinstance(sc, nn.Identity):
                ws, ps, bsum = None, None, b2
            else:
                _require(sc, modeling.ResNetShortCut)
                _require(sc.convolution, nn.Conv2d)
                _require(sc.normalization, nn.BatchNorm2d)
                ws, bs = _fold(sc.convolution, sc.normalization)
                ps = _conv_params(sc.convolution)
                bsum = (b2 + bs).contiguous()
            steps.append(
                {
                    "kind": "basic",
                    "w1": w1,
                    "b1": b1,
                    "p1": _conv_params(c1.convolution),
                    "w2": w2,
                    "bsum": bsum,
                    "p2": _conv_params(c2.convolution),
                    "ws": ws,
                    "ps": ps,
                }
            )
    classifier = module.classifier
    _require(classifier, nn.Sequential)
    if len(classifier) != 2 or not isinstance(classifier[0], nn.Flatten) or not isinstance(classifier[1], nn.Linear):
        raise ValueError("unexpected classifier structure")
    lin = classifier[1]
    if lin.bias is None:
        raise ValueError("classifier linear has no bias")
    return {
        "steps": steps,
        "pool_size": _pair(resnet.pooler.output_size),
        "lin_w": lin.weight.detach().clone().contiguous(),
        "lin_b": lin.bias.detach().clone().contiguous(),
        "num_channels": int(emb.num_channels),
        "dtype": stem.convolution.weight.dtype,
    }


class _Pointwise:
    def __init__(self):
        self.launch = None
        self.broken = False

    def _compile(self):
        if self.launch is not None or self.broken:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _bias_relu(x_ptr, b_ptr, y_ptr, HW, C, BLOCK: tl.constexpr):
            plane = tl.program_id(0)
            off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            m = off < HW
            base = plane * HW
            xv = tl.load(x_ptr + base + off, mask=m, other=0.0)
            bv = tl.load(b_ptr + plane % C)
            tl.store(y_ptr + base + off, tl.maximum(xv + bv, 0.0), mask=m)

        @triton.jit
        def _add_bias_relu(a_ptr, s_ptr, b_ptr, y_ptr, HW, C, BLOCK: tl.constexpr):
            plane = tl.program_id(0)
            off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            m = off < HW
            base = plane * HW
            av = tl.load(a_ptr + base + off, mask=m, other=0.0)
            sv = tl.load(s_ptr + base + off, mask=m, other=0.0)
            bv = tl.load(b_ptr + plane % C)
            tl.store(y_ptr + base + off, tl.maximum(av + sv + bv, 0.0), mask=m)

        @triton.jit
        def _bias_relu_pool3x3s2p1(x_ptr, b_ptr, y_ptr, H, W, OH, OW, C, BLOCK: tl.constexpr):
            # Fused per-channel bias + ReLU + 3x3/stride-2/pad-1 max pool
            # (the installed transformers stem pooler geometry). Reads the raw
            # conv output once and writes only the pooled tensor:
            # relu(max(x_ij) + b) == max(relu(x_ij + b)) for finite floats, so
            # the selected element and per-element sums match the unfused
            # accepted path bitwise. Padded border lanes load -inf, exactly the
            # identity element of max pooling.
            plane = tl.program_id(0)
            row = tl.program_id(1)
            col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
            mout = col < OW
            base = plane * H * W
            h0 = row * 2 - 1
            w0 = col * 2 - 1
            best = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
            for dr in tl.static_range(3):
                rvalid = (h0 + dr >= 0) & (h0 + dr < H)
                for dc in tl.static_range(3):
                    cvalid = (w0 + dc >= 0) & (w0 + dc < W)
                    mask = mout & rvalid & cvalid
                    v = tl.load(
                        x_ptr + base + (h0 + dr) * W + (w0 + dc),
                        mask=mask,
                        other=float("-inf"),
                    )
                    best = tl.maximum(best, v)
            bv = tl.load(b_ptr + plane % C)
            out = tl.maximum(best + bv, 0.0)
            dst = (plane * OH + row) * OW + col
            tl.store(y_ptr + dst, out, mask=mout)

        def launch(kind, a, s, b, out, hw, planes, c):
            grid = (planes, (hw + 1023) // 1024)
            if kind == "bias":
                _bias_relu[grid](a, b, out, hw, c, BLOCK=1024, num_warps=4)
            else:
                _add_bias_relu[grid](a, s, b, out, hw, c, BLOCK=1024, num_warps=4)

        def launch_pool(x, b, out, h, w, oh, ow, planes, c):
            grid = (planes, oh, (ow + 63) // 64)
            _bias_relu_pool3x3s2p1[grid](x, b, out, h, w, oh, ow, c, BLOCK=64, num_warps=2)

        self.launch = launch
        self.launch_pool = launch_pool

    def bias_relu(self, x, b):
        if x.is_cuda and x.is_contiguous() and not self.broken:
            try:
                self._compile()
                n, c, h, w = x.shape
                out = torch.empty_like(x)
                self.launch("bias", x, None, b, out, h * w, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken = True
        return torch.relu(x + b.view(1, -1, 1, 1))

    def add_bias_relu(self, a, s, b):
        if (
            a.is_cuda
            and a.is_contiguous()
            and s.is_contiguous()
            and not self.broken
        ):
            try:
                self._compile()
                n, c, h, w = a.shape
                out = torch.empty_like(a)
                self.launch("add", a, s, b, out, h * w, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken = True
        return torch.relu(a + s + b.view(1, -1, 1, 1))

    def bias_relu_pool(self, x, b, k, s, p, d, ceil):
        # Only the exact stem geometry installed by transformers
        # (MaxPool2d(kernel_size=3, stride=2, padding=1), ceil_mode=False) is
        # fused; any other case keeps the accepted bundle's separate
        # relu(x+b) then F.max_pool2d sequence bitwise.
        if (
            x.is_cuda
            and x.is_contiguous()
            and not self.broken
            and k == (3, 3)
            and s == (2, 2)
            and p == (1, 1)
            and d == (1, 1)
            and not ceil
        ):
            n, c, h, w = x.shape
            # PyTorch floor output size for this geometry:
            # floor((H + 2*pad - dil*(k-1) - 1)/stride) + 1 = (H-1)//2 + 1
            oh = (h - 1) // 2 + 1
            ow = (w - 1) // 2 + 1
            try:
                self._compile()
                out = torch.empty(
                    (n, c, oh, ow), dtype=x.dtype, device=x.device
                )
                self.launch_pool(x, b, out, h, w, oh, ow, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken = True
        x = torch.relu(x + b.view(1, -1, 1, 1))
        return F.max_pool2d(x, k, s, p, d, ceil)


def _run_eager(plan, x, pw):
    for st in plan["steps"]:
        kind = st["kind"]
        if kind == "stem":
            s, p, d, g = st["p"]
            x = F.conv2d(x, st["w"], None, stride=s, padding=p, dilation=d, groups=g)
            x = pw.bias_relu(x, st["b"])
        elif kind == "stem_pool":
            s, p, d, g = st["p"]
            x = F.conv2d(x, st["w"], None, stride=s, padding=p, dilation=d, groups=g)
            x = pw.bias_relu_pool(x, st["b"], st["k"], st["s"], st["pp"], st["d"], st["ceil"])
        elif kind == "maxpool":
            x = F.max_pool2d(x, st["k"], st["s"], st["p"], st["d"], st["ceil"])
        else:
            s1, p1, d1, g1 = st["p1"]
            h = F.conv2d(x, st["w1"], None, stride=s1, padding=p1, dilation=d1, groups=g1)
            h = pw.bias_relu(h, st["b1"])
            s2, p2, d2, g2 = st["p2"]
            h2 = F.conv2d(h, st["w2"], None, stride=s2, padding=p2, dilation=d2, groups=g2)
            if st["ws"] is not None:
                ss, ps, ds, gs = st["ps"]
                sres = F.conv2d(x, st["ws"], None, stride=ss, padding=ps, dilation=ds, groups=gs)
            else:
                sres = x
            x = pw.add_bias_relu(h2, sres, st["bsum"])
    x = F.adaptive_avg_pool2d(x, plan["pool_size"])
    x = torch.flatten(x, 1)
    return F.linear(x, plan["lin_w"], plan["lin_b"])


def _sig(x):
    return (
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        str(x.device),
        x.data_ptr(),
    )


class _FusedForward:
    def __init__(self, plan, parent, route, call_id):
        self.plan = plan
        self.parent = parent
        self.route = route
        self.call_id = call_id
        self.pw = _Pointwise()
        self.cuda_calls = 0
        self.graph = None
        self.graph_logits = None
        self.graph_sig = None
        self.graph_disabled = False
        self.eager_label = (
            "eager_fused_stem_pool"
            if any(st["kind"] == "stem_pool" for st in plan["steps"])
            else "eager_fused"
        )

    def _record(self, label):
        try:
            self.route(self.call_id, label)
        except Exception:
            pass

    def _capture(self, x, logits):
        try:
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream(device=x.device)
            side.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(side):
                _run_eager(self.plan, x, self.pw)
            torch.cuda.current_stream(x.device).wait_stream(side)
            with torch.cuda.graph(graph):
                self.graph_logits = _run_eager(self.plan, x, self.pw)
            self.graph = graph
            self.graph_sig = _sig(x)
            graph.replay()
            torch.cuda.synchronize(x.device)
            if not torch.allclose(
                self.graph_logits.detach(), logits.detach(), atol=1e-4, rtol=1e-4
            ):
                self.graph = None
                self.graph_sig = None
                self.graph_logits = None
                self.graph_disabled = True
                self._record("capture_verification_failed")
            else:
                self._record("captured")
        except Exception:
            self.graph = None
            self.graph_sig = None
            self.graph_logits = None
            self.graph_disabled = True
            self._record("capture_failed")

    def __call__(self, *args, **kwargs):
        pv = kwargs.get("pixel_values")
        if pv is None and len(args) >= 1:
            pv = args[0]
        labels = kwargs.get("labels")
        if labels is None and len(args) >= 2:
            labels = args[1]
        ohs = kwargs.get("output_hidden_states")
        if ohs is None and len(args) >= 3:
            ohs = args[2]
        rd = kwargs.get("return_dict")
        if rd is None and len(args) >= 4:
            rd = args[3]
        extra = {
            k: v
            for k, v in kwargs.items()
            if k not in ("pixel_values", "labels", "output_hidden_states", "return_dict")
        }
        clean = (
            isinstance(pv, torch.Tensor)
            and pv.dim() == 4
            and pv.dtype == self.plan["dtype"]
            and labels is None
            and not ohs
            and not extra
            and rd in (None, True)
            and pv.shape[1] == self.plan["num_channels"]
        )
        if not clean:
            self._record("parent_fallback_unsupported_signature")
            return self.parent(*args, **kwargs)
        x = pv if pv.is_contiguous() else pv.contiguous()
        sig = _sig(x)
        if self.graph is not None and sig == self.graph_sig:
            try:
                self.graph.replay()
                out = self.graph_logits.clone()
                self._record("cuda_graph_replay")
                return _Output(loss=None, logits=out, hidden_states=None)
            except Exception:
                self.graph = None
                self.graph_sig = None
                self.graph_logits = None
                self.graph_disabled = True
                self._record("graph_replay_failed")
        logits = _run_eager(self.plan, x, self.pw)
        if x.is_cuda:
            self.cuda_calls += 1
            if self.cuda_calls == 1 and self.graph is None and not self.graph_disabled:
                self._capture(x, logits)
        self._record(self.eager_label)
        return _Output(loss=None, logits=logits, hidden_states=None)


def _setup_failure(parent, route, call_id, reason):
    def _call(*args, **kwargs):
        try:
            route(call_id, "setup_failure:" + reason)
        except Exception:
            pass
        return parent(*args, **kwargs)

    return _call


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    modules = context.get("modules") or {}
    references = context.get("references") or {}
    replacements = {}
    module = modules.get(CALL_ID)
    parent = references.get(CALL_ID)
    if parent is None:
        return replacements
    if module is None:
        replacements[CALL_ID] = _setup_failure(parent, record_route, CALL_ID, "missing_module")
        return replacements
    try:
        with torch.inference_mode(False), torch.no_grad():
            plan = _build_plan(module)
    except Exception:
        replacements[CALL_ID] = _setup_failure(parent, record_route, CALL_ID, "plan_build")
        return replacements
    replacements[CALL_ID] = _FusedForward(plan, parent, record_route, CALL_ID)
    return replacements
