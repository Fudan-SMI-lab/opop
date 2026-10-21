"""Whole-bundle fused FP32 inference replacement for the ResNetForImageClassification
module call f2cf62caf57ad09f5d2f36aa, composing on the installed bundle
trial-a9f72afa23-param-f2e37a0c (accepted at 3.4017685 ms), which this candidate
explicitly replaces and whose entire behavior it re-owns: eval BatchNorm2d folded
into bias-free conv weights, fused residual add+bias+ReLU and conv bias+ReLU Triton
pointwise kernels, identical aten conv2d/adaptive_avg_pool2d/linear ops under the
frozen FP32/no-TF32/cudnn.benchmark=False policy, lazy side-stream warmup ->
CUDA-graph capture -> verified replay with input-signature guards, eager fused path
and parent fallback for unsupported signatures.

New in this revision (all inside the accepted lazy capture/warmup/verified-replay
flow, each piece independently guarded back to the accepted bundle's bitwise aten
path):

1. Fused shortcut epilogue for the three downsample stages: one Triton kernel per
   stage computes relu(conv2_out + 1x1/stride-2 projection(x) + bsum) with the
   K=64/128/256 projection accumulated in-register via tl.dot
   (input_precision="ieee"), deleting the sres tensor round trip, the three
   implicit_convolve_sgemm kernels, their computeOffsetsKernel instances and the
   three separate _add_bias_relu kernels (measured 120.1us total on the accepted
   revision).
2. A Triton implicit-GEMM conv+bias+ReLU for layer4 block1 conv1 (3x3/stride-2,
   256->512, 7x7 output) whose 7x7 output grid is tile-quantized in cudnn
   (152.8us conv + 9.4us epilogue at 24.2 TF/s).
3. A Triton implicit-GEMM replacement for the stem 7x7/stride-2 conv
   (257.7us at 28.9 TF/s) feeding the unchanged fused pool epilogue.
4. A retiled fused bias+ReLU+3x3/stride-2/pad-1 max-pool kernel with cross-row
   register reuse (R=8 output rows per program), replacing the accepted
   one-row-per-program kernel that ran at 1.03TB/s (124.7us).
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
    stem_p = _conv_params(stem.convolution)
    fused_stem_conv = (
        tuple(stem.convolution.weight.shape) == (64, 3, 7, 7)
        and stem_p == ((2, 2), (3, 3), (1, 1), 1)
    )
    if k == (3, 3) and s == (2, 2) and p == (1, 1) and d == (1, 1) and not ceil:
        steps.append(
            {
                "kind": "stem_pool",
                "w": w,
                "b": b,
                "p": stem_p,
                "k": k,
                "s": s,
                "pp": p,
                "d": d,
                "ceil": ceil,
                "fused_conv": fused_stem_conv,
            }
        )
    else:
        steps.append({"kind": "stem", "w": w, "b": b, "p": stem_p})
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
            p1 = _conv_params(c1.convolution)
            conv1_fused = (
                tuple(c1.convolution.weight.shape) == (512, 256, 3, 3)
                and p1 == ((2, 2), (1, 1), (1, 1), 1)
            )
            sc = layer.shortcut
            if isinstance(sc, nn.Identity):
                ws, ps, bsum = None, None, b2
                sc_fused = False
            else:
                _require(sc, modeling.ResNetShortCut)
                _require(sc.convolution, nn.Conv2d)
                _require(sc.normalization, nn.BatchNorm2d)
                ws, bs = _fold(sc.convolution, sc.normalization)
                ps = _conv_params(sc.convolution)
                bsum = (b2 + bs).contiguous()
                sc_fused = (
                    ws.shape[2:] == (1, 1)
                    and ps == ((2, 2), (0, 0), (1, 1), 1)
                    and ws.shape[0] % 128 == 0
                    and ws.shape[1] % 64 == 0
                )
            steps.append(
                {
                    "kind": "basic",
                    "w1": w1,
                    "b1": b1,
                    "p1": p1,
                    "w2": w2,
                    "bsum": bsum,
                    "p2": _conv_params(c2.convolution),
                    "ws": ws,
                    "ps": ps,
                    "conv1_fused": conv1_fused,
                    "sc_fused": sc_fused,
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


class _Kernels:
    def __init__(self):
        self.pw = None
        self.pool = None
        self.stem = None
        self.conv3 = None
        self.proj = None
        self.broken_pw = False
        self.broken_pool = False
        self.broken_stem = False
        self.broken_conv3 = False
        self.broken_proj = False

    def _ensure_pw(self):
        if self.pw is not None or self.broken_pw:
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

        self.k_bias_relu = _bias_relu
        self.k_add_bias_relu = _add_bias_relu

        def launch(kind, a, s, b, out, hw, planes, c):
            grid = (planes, (hw + 1023) // 1024)
            if kind == "bias":
                _bias_relu[grid](a, b, out, hw, c, BLOCK=1024, num_warps=4)
            else:
                _add_bias_relu[grid](a, s, b, out, hw, c, BLOCK=1024, num_warps=4)

        self.pw = launch

    def _ensure_pool(self):
        if self.pool is not None or self.broken_pool:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _rowmax3(x_ptr, rowbase, col, m, rv, W):
            o = rowbase + 2 * col - 1
            a = tl.load(x_ptr + o, mask=m & rv & (col > 0), other=float("-inf"))
            b = tl.load(x_ptr + o + 1, mask=m & rv, other=float("-inf"))
            c = tl.load(x_ptr + o + 2, mask=m & rv & ((2 * col + 1) < W), other=float("-inf"))
            return tl.maximum(tl.maximum(a, b), c)

        @triton.jit
        def _bias_relu_pool_rows(x_ptr, b_ptr, y_ptr, H, W, OH, OW, C,
                                 R: tl.constexpr, BW: tl.constexpr):
            plane = tl.program_id(0)
            r0 = tl.program_id(1) * R
            c0 = tl.program_id(2) * BW
            col = c0 + tl.arange(0, BW)
            mout = col < OW
            base = plane * H * W
            bv = tl.load(b_ptr + plane % C)
            im = 2 * r0 - 1
            mprev = _rowmax3(x_ptr, base + im * W, col, mout, (im >= 0) & (im < H), W)
            for rr in tl.static_range(R):
                r = r0 + rr
                icur = 2 * r
                mcur = _rowmax3(x_ptr, base + icur * W, col, mout, icur < H, W)
                nxt = 2 * r + 1
                mnext = _rowmax3(x_ptr, base + nxt * W, col, mout, nxt < H, W)
                best = tl.maximum(tl.maximum(mprev, mcur), mnext)
                out = tl.maximum(best + bv, 0.0)
                tl.store(y_ptr + (plane * OH + r) * OW + col, out, mask=mout & (r < OH))
                mprev = mnext

        self.k_pool = _bias_relu_pool_rows

        def launch(x, b, out, h, w, oh, ow, planes, c):
            grid = (planes, (oh + 7) // 8, (ow + 63) // 64)
            _bias_relu_pool_rows[grid](x, b, out, h, w, oh, ow, c, R=8, BW=64, num_warps=2)

        self.pool = launch

    def _ensure_stem(self):
        if self.stem is not None or self.broken_stem:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _conv7x7s2(x_ptr, w_ptr, y_ptr, Cin, H, W, OH, OW, HW, XP, L,
                       BSP: tl.constexpr):
            co = tl.arange(0, 64)
            l = tl.program_id(0) * BSP + tl.arange(0, BSP)
            lm = l < L
            n = l // HW
            p = l % HW
            ph = p // OW
            pw = p % OW
            ybase = n * (64 * HW) + p
            acc = tl.zeros((64, BSP), dtype=tl.float32)
            for k0 in tl.static_range(0, 128, 64):
                k = k0 + tl.arange(0, 64)
                ci = k // 49
                r = k % 49
                kh = r // 7
                kw = r % 7
                ih = 2 * ph[None, :] + kh[:, None] - 3
                iw = 2 * pw[None, :] + kw[:, None] - 3
                mm = lm[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                a = tl.load(w_ptr + co[:, None] * 147 + k[None, :])
                b = tl.load(x_ptr + n[None, :] * (Cin * XP) + ci[:, None] * XP + ih * W + iw, mask=mm, other=0.0)
                acc = tl.dot(a, b, acc, input_precision="ieee")
            k = 128 + tl.arange(0, 32)
            km = k < 147
            ci = k // 49
            r = k % 49
            kh = r // 7
            kw = r % 7
            ih = 2 * ph[None, :] + kh[:, None] - 3
            iw = 2 * pw[None, :] + kw[:, None] - 3
            mm = km[:, None] & lm[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            a = tl.load(w_ptr + co[:, None] * 147 + k[None, :], mask=km[None, :], other=0.0)
            b = tl.load(x_ptr + n[None, :] * (Cin * XP) + ci[:, None] * XP + ih * W + iw, mask=mm, other=0.0)
            acc = tl.dot(a, b, acc, input_precision="ieee")
            tl.store(y_ptr + ybase[None, :] + co[:, None] * HW, acc, mask=lm[None, :])

        self.k_stem = _conv7x7s2

        def launch(x, w2d, y, cin, h, w, oh, ow, hw, xp, ell):
            _conv7x7s2[((ell + 127) // 128,)](x, w2d, y, cin, h, w, oh, ow, hw, xp, ell, BSP=128, num_warps=8)

        self.stem = launch

    def _ensure_conv3(self):
        if self.conv3 is not None or self.broken_conv3:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _conv3x3s2(x_ptr, w_ptr, b_ptr, y_ptr, Cin, Cout, K, H, W, OH, OW, HW, XP, L,
                       BCO: tl.constexpr, BSP: tl.constexpr, BK: tl.constexpr):
            co = tl.program_id(0) * BCO + tl.arange(0, BCO)
            l = tl.program_id(1) * BSP + tl.arange(0, BSP)
            lm = l < L
            n = l // HW
            p = l % HW
            ph = p // OW
            pw = p % OW
            ybase = n * (Cout * HW) + p
            acc = tl.zeros((BCO, BSP), dtype=tl.float32)
            for k0 in range(0, K, BK):
                k = k0 + tl.arange(0, BK)
                ci = k // 9
                r = k % 9
                kh = r // 3
                kw = r % 3
                ih = 2 * ph[None, :] + kh[:, None] - 1
                iw = 2 * pw[None, :] + kw[:, None] - 1
                mm = lm[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                a = tl.load(w_ptr + co[:, None] * K + k[None, :])
                b = tl.load(x_ptr + n[None, :] * (Cin * XP) + ci[:, None] * XP + ih * W + iw, mask=mm, other=0.0)
                acc = tl.dot(a, b, acc, input_precision="ieee")
            bv = tl.load(b_ptr + co, mask=co < Cout, other=0.0)
            out = tl.maximum(acc + bv[:, None], 0.0)
            tl.store(y_ptr + ybase[None, :] + co[:, None] * HW, out,
                     mask=(co < Cout)[:, None] & lm[None, :])

        self.k_conv3 = _conv3x3s2

        def launch(x, w2d, b, y, cin, cout, kk, h, w, oh, ow, hw, xp, ell):
            grid = ((cout + 127) // 128, (ell + 63) // 64)
            _conv3x3s2[grid](x, w2d, b, y, cin, cout, kk, h, w, oh, ow, hw, xp, ell,
                             BCO=128, BSP=64, BK=64, num_warps=8)

        self.conv3 = launch

    def _ensure_proj(self):
        if self.proj is not None or self.broken_proj:
            return
        import triton
        import triton.language as tl

        @triton.jit
        def _proj_add(x_ptr, h_ptr, w_ptr, b_ptr, y_ptr,
                      Cin, Cout, HW, W, XP, L,
                      BCO: tl.constexpr, BSP: tl.constexpr, BK: tl.constexpr):
            co = tl.program_id(0) * BCO + tl.arange(0, BCO)
            l = tl.program_id(1) * BSP + tl.arange(0, BSP)
            lm = l < L
            n = l // HW
            q = l % HW
            hrow = q // W
            wcol = q % W
            xoff = n * (Cin * XP) + 4 * hrow * W + 2 * wcol
            ybase = n * (Cout * HW) + q
            acc = tl.zeros((BCO, BSP), dtype=tl.float32)
            for k0 in range(0, Cin, BK):
                k = k0 + tl.arange(0, BK)
                a = tl.load(w_ptr + co[:, None] * Cin + k[None, :])
                b = tl.load(x_ptr + xoff[None, :] + k[:, None] * XP, mask=lm[None, :], other=0.0)
                acc = tl.dot(a, b, acc, input_precision="ieee")
            com = co < Cout
            h2 = tl.load(h_ptr + ybase[None, :] + co[:, None] * HW,
                         mask=com[:, None] & lm[None, :], other=0.0)
            bv = tl.load(b_ptr + co, mask=com, other=0.0)
            out = tl.maximum(acc + h2 + bv[:, None], 0.0)
            tl.store(y_ptr + ybase[None, :] + co[:, None] * HW, out,
                     mask=com[:, None] & lm[None, :])

        self.k_proj = _proj_add

        def launch(x, h2, w2d, b, y, cin, cout, hw, w, xp, ell):
            grid = ((cout + 127) // 128, (ell + 63) // 64)
            _proj_add[grid](x, h2, w2d, b, y, cin, cout, hw, w, xp, ell,
                            BCO=128, BSP=64, BK=64, num_warps=8)

        self.proj = launch

    def bias_relu(self, x, b):
        if x.is_cuda and x.is_contiguous() and not self.broken_pw:
            try:
                self._ensure_pw()
                n, c, h, w = x.shape
                out = torch.empty_like(x)
                self.pw("bias", x, None, b, out, h * w, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_pw = True
        return torch.relu(x + b.view(1, -1, 1, 1))

    def add_bias_relu(self, a, s, b):
        if (
            a.is_cuda
            and a.is_contiguous()
            and s.is_contiguous()
            and not self.broken_pw
        ):
            try:
                self._ensure_pw()
                n, c, h, w = a.shape
                out = torch.empty_like(a)
                self.pw("add", a, s, b, out, h * w, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_pw = True
        return torch.relu(a + s + b.view(1, -1, 1, 1))

    def bias_relu_pool(self, x, b, k, s, p, d, ceil):
        if (
            x.is_cuda
            and x.is_contiguous()
            and not self.broken_pool
            and k == (3, 3)
            and s == (2, 2)
            and p == (1, 1)
            and d == (1, 1)
            and not ceil
        ):
            n, c, h, w = x.shape
            oh = (h - 1) // 2 + 1
            ow = (w - 1) // 2 + 1
            try:
                self._ensure_pool()
                out = torch.empty((n, c, oh, ow), dtype=x.dtype, device=x.device)
                self.pool(x, b, out, h, w, oh, ow, n * c, c)
                return out
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_pool = True
        x = torch.relu(x + b.view(1, -1, 1, 1))
        return F.max_pool2d(x, k, s, p, d, ceil)

    def stem7x7s2(self, x, w, s, p, d, g):
        if (
            not self.broken_stem
            and x.is_cuda
            and x.is_contiguous()
            and w.is_contiguous()
            and x.dtype == torch.float32
            and x.dim() == 4
            and x.shape[1] == 3
            and s == (2, 2)
            and p == (3, 3)
            and d == (1, 1)
            and g == 1
            and tuple(w.shape) == (64, 3, 7, 7)
        ):
            try:
                self._ensure_stem()
                n, _, h, w_in = x.shape
                oh = (h - 1) // 2 + 1
                ow = (w_in - 1) // 2 + 1
                y = torch.empty((n, 64, oh, ow), dtype=x.dtype, device=x.device)
                ell = n * oh * ow
                self.stem(x, w.view(64, 147), y, 3, h, w_in, oh, ow, oh * ow, h * w_in, ell)
                return y
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_stem = True
        return F.conv2d(x, w, None, stride=s, padding=p, dilation=d, groups=g)

    def conv3x3s2_bias_relu(self, x, w, b, s, p, d, g):
        if (
            not self.broken_conv3
            and x.is_cuda
            and x.is_contiguous()
            and w.is_contiguous()
            and b.is_contiguous()
            and x.dtype == torch.float32
            and x.dim() == 4
            and w.dim() == 4
            and w.shape[2:] == (3, 3)
            and x.shape[1] == w.shape[1]
            and s == (2, 2)
            and p == (1, 1)
            and d == (1, 1)
            and g == 1
            and w.shape[0] % 128 == 0
            and w.shape[1] % 64 == 0
            and (w.shape[1] * 9) % 64 == 0
        ):
            try:
                self._ensure_conv3()
                n, _, h, w_in = x.shape
                cout, cin = w.shape[0], w.shape[1]
                oh = (h - 1) // 2 + 1
                ow = (w_in - 1) // 2 + 1
                y = torch.empty((n, cout, oh, ow), dtype=x.dtype, device=x.device)
                ell = n * oh * ow
                self.conv3(x, w.view(cout, cin * 9), b, y, cin, cout, cin * 9,
                           h, w_in, oh, ow, oh * ow, h * w_in, ell)
                return y
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_conv3 = True
        h = F.conv2d(x, w, None, stride=s, padding=p, dilation=d, groups=g)
        return torch.relu(h + b.view(1, -1, 1, 1))

    def proj_add_bias_relu(self, x, h2, ws, bsum):
        if (
            not self.broken_proj
            and x.is_cuda
            and x.is_contiguous()
            and h2.is_contiguous()
            and ws.is_contiguous()
            and bsum.is_contiguous()
            and x.dtype == torch.float32
            and h2.dtype == torch.float32
            and x.dim() == 4
            and h2.dim() == 4
            and ws.dim() == 4
            and ws.shape[2:] == (1, 1)
            and x.shape[0] == h2.shape[0]
            and x.shape[1] == ws.shape[1]
            and h2.shape[1] == ws.shape[0]
            and x.shape[2] == 2 * h2.shape[2]
            and x.shape[3] == 2 * h2.shape[3]
            and ws.shape[0] % 128 == 0
            and ws.shape[1] % 64 == 0
        ):
            try:
                self._ensure_proj()
                n, cin, _, _ = x.shape
                cout = h2.shape[1]
                hw = h2.shape[2] * h2.shape[3]
                y = torch.empty_like(h2)
                ell = n * hw
                self.proj(x, h2, ws.view(cout, cin), bsum, y, cin, cout, hw,
                          h2.shape[3], 4 * hw, ell)
                return y
            except Exception:
                if torch.cuda.is_current_stream_capturing():
                    raise
                self.broken_proj = True
        sres = F.conv2d(x, ws, None, stride=2)
        return torch.relu(h2 + sres + bsum.view(1, -1, 1, 1))


def _run_eager(plan, x, kw):
    for st in plan["steps"]:
        kind = st["kind"]
        if kind == "stem":
            s, p, d, g = st["p"]
            x = F.conv2d(x, st["w"], None, stride=s, padding=p, dilation=d, groups=g)
            x = kw.bias_relu(x, st["b"])
        elif kind == "stem_pool":
            s, p, d, g = st["p"]
            if st.get("fused_conv"):
                x = kw.stem7x7s2(x, st["w"], s, p, d, g)
            else:
                x = F.conv2d(x, st["w"], None, stride=s, padding=p, dilation=d, groups=g)
            x = kw.bias_relu_pool(x, st["b"], st["k"], st["s"], st["pp"], st["d"], st["ceil"])
        elif kind == "maxpool":
            x = F.max_pool2d(x, st["k"], st["s"], st["p"], st["d"], st["ceil"])
        else:
            s1, p1, d1, g1 = st["p1"]
            if st.get("conv1_fused"):
                h = kw.conv3x3s2_bias_relu(x, st["w1"], st["b1"], s1, p1, d1, g1)
            else:
                h = F.conv2d(x, st["w1"], None, stride=s1, padding=p1, dilation=d1, groups=g1)
                h = kw.bias_relu(h, st["b1"])
            s2, p2, d2, g2 = st["p2"]
            h2 = F.conv2d(h, st["w2"], None, stride=s2, padding=p2, dilation=d2, groups=g2)
            if st["ws"] is not None:
                if st.get("sc_fused"):
                    x = kw.proj_add_bias_relu(x, h2, st["ws"], st["bsum"])
                else:
                    ss, ps, ds, gs = st["ps"]
                    sres = F.conv2d(x, st["ws"], None, stride=ss, padding=ps, dilation=ds, groups=gs)
                    x = kw.add_bias_relu(h2, sres, st["bsum"])
            else:
                x = kw.add_bias_relu(h2, x, st["bsum"])
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
        self.kw = _Kernels()
        self.cuda_calls = 0
        self.graph = None
        self.graph_logits = None
        self.graph_sig = None
        self.graph_disabled = False
        self.eager_label = "eager_fused_v3"

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
                _run_eager(self.plan, x, self.kw)
            torch.cuda.current_stream(x.device).wait_stream(side)
            with torch.cuda.graph(graph):
                self.graph_logits = _run_eager(self.plan, x, self.kw)
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
        logits = _run_eager(self.plan, x, self.kw)
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
