"""Candidate for project p6-resnet-throughput-a, region eca3643289778aea5af3.

Replaces the single observed ResNetForImageClassification invocation
(call 80ff8693d7ac3cf2433b10f2) of the frozen offline batch-32 workload with a
fused execution of the identical serial forward graph:

  * every convolution, the stem max-pool, the adaptive average pool and the
    classifier linear keep their exact eager FP32 kernels (TF32 stays off),
  * eval-mode BatchNorm2d is a pure per-channel affine map of frozen running
    statistics, so BN+ReLU collapses into one custom pointwise kernel and the
    residual add + ReLU epilogue (including the projection shortcut's BN)
    collapses into a second custom pointwise kernel,
  * the whole region is captured once into a CUDA graph keyed on the live
    workload input buffer, so each invocation replays the full kernel sequence
    (reads current buffer contents, recomputes and clones fresh logits) with no
    per-module dispatch overhead.

Every invocation executes the complete workload; no outputs are cached. Any
signature that does not match the observed one (other batch, kwargs, devices,
dtypes, strides) falls back to the parent bound forward.
"""

from __future__ import annotations

import os
import tempfile
import threading

import torch
import torch.nn.functional as F

REGION_CALL_ID = "80ff8693d7ac3cf2433b10f2"

_OBSERVED_SHAPE = (32, 3, 224, 224)
_OBSERVED_STRIDE = (150528, 50176, 224, 1)

_BUILD_GUARD = threading.Lock()
_EXT = None
_EXT_TRIED = False

_EXT_NAME = "opop_resnet18_fused_v1"

_CPP_SRC = r"""
#include <torch/extension.h>
at::Tensor bn_relu(at::Tensor x, at::Tensor s, at::Tensor t);
at::Tensor bn_add_relu(at::Tensor m, at::Tensor s2, at::Tensor t2,
                       at::Tensor r, at::Tensor sr, at::Tensor tr);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kThreads = 256;

__global__ void bn_relu_kernel(const float* __restrict__ x,
                               const float* __restrict__ s,
                               const float* __restrict__ t,
                               float* __restrict__ y,
                               const int64_t inner,
                               const int C) {
    const int plane = static_cast<int>(blockIdx.x);
    const int ch = plane % C;
    const float sc = s[ch];
    const float sh = t[ch];
    const int64_t base = static_cast<int64_t>(plane) * inner;
    const float* xp = x + base;
    float* yp = y + base;
    const int64_t span = static_cast<int64_t>(blockDim.x) * gridDim.y;
    for (int64_t i = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.y) * blockDim.x;
         i < inner; i += span) {
        const float v = fmaf(xp[i], sc, sh);
        yp[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void bn_add_relu_kernel(const float* __restrict__ m,
                                   const float* __restrict__ s2,
                                   const float* __restrict__ t2,
                                   const float* __restrict__ r,
                                   const float* __restrict__ sr,
                                   const float* __restrict__ tr,
                                   float* __restrict__ y,
                                   const int64_t inner,
                                   const int C) {
    const int plane = static_cast<int>(blockIdx.x);
    const int ch = plane % C;
    const float as = s2[ch];
    const float at = t2[ch];
    const float rs = sr[ch];
    const float rt = tr[ch];
    const int64_t base = static_cast<int64_t>(plane) * inner;
    const float* mp = m + base;
    const float* rp = r + base;
    float* yp = y + base;
    const int64_t span = static_cast<int64_t>(blockDim.x) * gridDim.y;
    for (int64_t i = static_cast<int64_t>(threadIdx.x) + static_cast<int64_t>(blockIdx.y) * blockDim.x;
         i < inner; i += span) {
        const float v = fmaf(mp[i], as, at) + fmaf(rp[i], rs, rt);
        yp[i] = v > 0.0f ? v : 0.0f;
    }
}

inline void check_args(const at::Tensor& x, const at::Tensor& y,
                       const at::Tensor& s, const char* who) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(),
                who, ": input must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(y.is_contiguous() && y.sizes() == x.sizes(),
                who, ": output must be contiguous with the input shape");
    TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.scalar_type() == at::kFloat &&
                    s.numel() == x.size(1),
                who, ": per-channel parameter must be contiguous CUDA float32 of size C");
}

inline dim3 plane_grid(const int64_t planes, const int64_t inner) {
    int64_t yblocks = (inner + kThreads - 1) / kThreads;
    if (yblocks < 1) yblocks = 1;
    if (yblocks > 65535) yblocks = 65535;
    TORCH_CHECK(planes <= 2147483647LL, "too many channel planes");
    return dim3(static_cast<unsigned int>(planes), static_cast<unsigned int>(yblocks));
}

}  // namespace

at::Tensor bn_relu(at::Tensor x, at::Tensor s, at::Tensor t) {
    const at::cuda::OptionalCUDAGuard guard(x.device());
    at::Tensor y = at::empty_like(x);
    check_args(x, y, s, "bn_relu");
    check_args(x, y, t, "bn_relu");
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t inner = x.size(2) * x.size(3);
    if (N * C * inner == 0) {
        return y;
    }
    const dim3 grid = plane_grid(N * C, inner);
    bn_relu_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.const_data_ptr<float>(), s.const_data_ptr<float>(), t.const_data_ptr<float>(),
        y.mutable_data_ptr<float>(), inner, static_cast<int>(C));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

at::Tensor bn_add_relu(at::Tensor m, at::Tensor s2, at::Tensor t2,
                       at::Tensor r, at::Tensor sr, at::Tensor tr) {
    const at::cuda::OptionalCUDAGuard guard(m.device());
    at::Tensor y = at::empty_like(m);
    check_args(m, y, s2, "bn_add_relu");
    check_args(m, y, t2, "bn_add_relu");
    check_args(m, y, sr, "bn_add_relu");
    TORCH_CHECK(r.is_cuda() && r.scalar_type() == at::kFloat && r.sizes() == m.sizes(),
                "bn_add_relu: residual must be a CUDA float32 tensor of the same shape");
    TORCH_CHECK(r.is_contiguous(), "bn_add_relu: residual must be contiguous");
    check_args(m, y, tr, "bn_add_relu");
    const int64_t N = m.size(0);
    const int64_t C = m.size(1);
    const int64_t inner = m.size(2) * m.size(3);
    if (N * C * inner == 0) {
        return y;
    }
    const dim3 grid = plane_grid(N * C, inner);
    bn_add_relu_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        m.const_data_ptr<float>(), s2.const_data_ptr<float>(), t2.const_data_ptr<float>(),
        r.const_data_ptr<float>(), sr.const_data_ptr<float>(), tr.const_data_ptr<float>(),
        y.mutable_data_ptr<float>(), inner, static_cast<int>(C));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""


def _load_ext():
    global _EXT, _EXT_TRIED
    if _EXT is not None:
        return _EXT
    with _BUILD_GUARD:
        if _EXT is not None:
            return _EXT
        if _EXT_TRIED:
            return None
        _EXT_TRIED = True
        try:
            from torch.utils.cpp_extension import load_inline

            build_dir = os.path.join(tempfile.gettempdir(), _EXT_NAME)
            os.makedirs(build_dir, exist_ok=True)
            _EXT = load_inline(
                name=_EXT_NAME,
                cpp_sources=[_CPP_SRC],
                cuda_sources=[_CUDA_SRC],
                functions=["bn_relu", "bn_add_relu"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                build_directory=build_dir,
                verbose=False,
            )
        except Exception:
            _EXT = None
    return _EXT


def _bn_affine(bn):
    """Fold frozen eval-mode BatchNorm2d into a per-channel affine map."""
    weight = bn.weight.detach()
    var = bn.running_var.detach()
    mean = bn.running_mean.detach()
    bias = bn.bias.detach() if bn.bias is not None else torch.zeros_like(mean)
    scale = weight * torch.rsqrt(var + bn.eps)
    shift = bias - mean * scale
    return scale.contiguous(), shift.contiguous()


def _build_fused(model, ext):
    """Precompute the invariant packed parameters of the fused forward."""
    from transformers.models.resnet.modeling_resnet import ResNetBasicLayer, ResNetShortCut

    resnet = model.resnet
    stem_layer = resnet.embedder.embedder
    stem_conv = stem_layer.convolution
    stem_s, stem_t = _bn_affine(stem_layer.normalization)
    stem_pool = resnet.embedder.pooler

    stem = {
        "w": stem_conv.weight.detach(),
        "b": stem_conv.bias.detach() if stem_conv.bias is not None else None,
        "stride": stem_conv.stride,
        "pad": stem_conv.padding,
        "dil": stem_conv.dilation,
        "s": stem_s,
        "t": stem_t,
    }

    layers = []
    for stage in resnet.encoder.stages:
        for layer in stage.layers:
            if not isinstance(layer, ResNetBasicLayer):
                raise NotImplementedError("fused path supports basic layers only")
            conv1 = layer.layer[0].convolution
            bn1 = layer.layer[0].normalization
            conv2 = layer.layer[1].convolution
            bn2 = layer.layer[1].normalization
            s1, t1 = _bn_affine(bn1)
            s2, t2 = _bn_affine(bn2)
            if isinstance(layer.shortcut, ResNetShortCut):
                sc_conv = layer.shortcut.convolution
                sr, tr = _bn_affine(layer.shortcut.normalization)
                proj = {
                    "w": sc_conv.weight.detach(),
                    "stride": sc_conv.stride,
                    "pad": sc_conv.padding,
                    "sr": sr,
                    "tr": tr,
                }
            else:
                proj = None
            c_out = conv2.out_channels
            device = conv2.weight.device
            layers.append(
                {
                    "w1": conv1.weight.detach(),
                    "b1": conv1.bias.detach() if conv1.bias is not None else None,
                    "stride1": conv1.stride,
                    "pad1": conv1.padding,
                    "dil1": conv1.dilation,
                    "s1": s1,
                    "t1": t1,
                    "w2": conv2.weight.detach(),
                    "b2": conv2.bias.detach() if conv2.bias is not None else None,
                    "pad2": conv2.padding,
                    "dil2": conv2.dilation,
                    "s2": s2,
                    "t2": t2,
                    "proj": proj,
                    "ones": torch.ones(c_out, device=device, dtype=torch.float32),
                    "zeros": torch.zeros(c_out, device=device, dtype=torch.float32),
                }
            )

    head = model.classifier[1]
    head_w = head.weight.detach()
    head_b = head.bias.detach() if head.bias is not None else None
    head_pool = resnet.pooler
    out_size = tuple(head_pool.output_size)

    def fused(x):
        h = F.conv2d(x, stem["w"], stem["b"], stride=stem["stride"], padding=stem["pad"], dilation=stem["dil"])
        h = ext.bn_relu(h, stem["s"], stem["t"])
        h = F.max_pool2d(
            h,
            kernel_size=stem_pool.kernel_size,
            stride=stem_pool.stride,
            padding=stem_pool.padding,
            dilation=stem_pool.dilation,
            ceil_mode=stem_pool.ceil_mode,
        )
        for lay in layers:
            a = F.conv2d(h, lay["w1"], lay["b1"], stride=lay["stride1"], padding=lay["pad1"], dilation=lay["dil1"])
            a = ext.bn_relu(a, lay["s1"], lay["t1"])
            m = F.conv2d(a, lay["w2"], lay["b2"], stride=(1, 1), padding=lay["pad2"], dilation=lay["dil2"])
            proj = lay["proj"]
            if proj is None:
                h = ext.bn_add_relu(m, lay["s2"], lay["t2"], h, lay["ones"], lay["zeros"])
            else:
                r = F.conv2d(h, proj["w"], None, stride=proj["stride"], padding=proj["pad"])
                h = ext.bn_add_relu(m, lay["s2"], lay["t2"], r, proj["sr"], proj["tr"])
        p = F.adaptive_avg_pool2d(h, output_size=out_size)
        p = torch.flatten(p, 1)
        return F.linear(p, head_w, head_b)

    return fused


def _build_mirror(model):
    """Parent-identical eager composite (same modules, same kernels)."""

    resnet = model.resnet

    def mirror(x):
        h = resnet.embedder(x)
        for stage in resnet.encoder.stages:
            h = stage(h)
        p = resnet.pooler(h)
        return model.classifier(p)

    return mirror


def _capture_graph(fn, x):
    """Warm up, then capture fn(x) into a CUDA graph keyed on the live buffer."""
    device = x.device
    torch.cuda.synchronize(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn(x)
    graph.replay()
    fresh = fn(x)
    torch.cuda.synchronize(device)
    max_diff = (static_out.detach() - fresh.detach()).abs().max().item()
    if not (max_diff <= 1e-5):
        raise RuntimeError(f"cuda graph replay mismatch: {max_diff}")
    return graph, static_out


def _try_build_engine(context):
    state = context["state"]
    model = context["models"]["model"]
    pixels = state.get("pixels")
    if pixels is None or not pixels.is_cuda or pixels.dtype is not torch.float32:
        return None
    if not torch.cuda.is_available():
        return None
    device = pixels.device
    engine = {"device": device, "graph": None, "static_out": None, "fn": None, "key": None}

    ext = _load_ext()
    compute = None
    fused_failed = False
    if ext is not None:
        try:
            fused = _build_fused(model, ext)
            with torch.inference_mode():
                ref_out = model(pixel_values=pixels)
                ref_logits = ref_out.logits.detach()
                fused_logits = fused(pixels)
            torch.cuda.synchronize(device)
            if not torch.allclose(fused_logits, ref_logits, atol=2e-5, rtol=2e-5):
                raise RuntimeError("fused numerics deviate from the parent forward")
            compute = fused
        except Exception:
            compute = None
            fused_failed = True
    if compute is None:
        compute = _build_mirror(model)

    engine["fn"] = compute
    engine["fused_available"] = compute is not None and not fused_failed

    try:
        with torch.inference_mode():
            graph, static_out = _capture_graph(compute, pixels)
        engine["graph"] = graph
        engine["static_out"] = static_out
        engine["key"] = (
            pixels.data_ptr(),
            tuple(pixels.shape),
            tuple(pixels.stride()),
            pixels.dtype,
            pixels.device.index,
        )
    except Exception:
        engine["graph"] = None
        engine["static_out"] = None
        engine["key"] = None
    return engine


def _matches_observed(x):
    return (
        torch.is_tensor(x)
        and x.dim() == 4
        and tuple(x.shape) == _OBSERVED_SHAPE
        and tuple(x.stride()) == _OBSERVED_STRIDE
        and x.dtype is torch.float32
        and x.is_cuda
    )


def build(context):
    references = context.get("references") or {}
    state = context.get("state") or {}
    model = (context.get("models") or {}).get("model") or state.get("model")
    parent_forward = references.get(REGION_CALL_ID) or model.forward

    try:
        engine = _try_build_engine(context)
    except Exception:
        engine = None

    def _output(logits):
        try:
            from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

            return ImageClassifierOutputWithNoAttention(loss=None, logits=logits, hidden_states=None)
        except Exception:

            class _Shim:
                __slots__ = ("loss", "logits", "hidden_states")

                def __init__(self, loss, logits, hidden_states):
                    self.loss = loss
                    self.logits = logits
                    self.hidden_states = hidden_states

            return _Shim(None, logits, None)

    def region(pixel_values=None, labels=None, output_hidden_states=None, return_dict=None, **kwargs):
        # Unobserved argument forms go straight to the parent bound forward.
        if kwargs or labels is not None:
            return parent_forward(
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        if output_hidden_states not in (None, False) or return_dict not in (None, True):
            return parent_forward(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        x = pixel_values
        if x is None or not _matches_observed(x):
            return parent_forward(pixel_values=x)

        if engine is not None and engine["fn"] is not None:
            try:
                with torch.inference_mode():
                    graph = engine["graph"]
                    if graph is not None and engine["key"] == (
                        x.data_ptr(),
                        tuple(x.shape),
                        tuple(x.stride()),
                        x.dtype,
                        x.device.index,
                    ):
                        graph.replay()
                        return _output(engine["static_out"].detach().clone())
                    logits = engine["fn"](x)
                return _output(logits)
            except Exception:
                pass
        return parent_forward(pixel_values=x)

    return {REGION_CALL_ID: region}
