"""Candidate: specialized direct 3x3 convolution kernel for the frozen ResNet-18
stage-4 conv2d sites (group 5621410346ca09a421d0).

Replaces the three observed aten.conv2d calls (1x512x7x7 input, 512x512x3x3
weight, padding 1, no bias) with a single specialized CUDA kernel:

* observed parent path: cudnn Winograd, 2 kernels per call, 67.4-67.9 us/call
  (56.0-56.2 us scudnn winograd launched as 16 CTAs of 256 threads for a
  512x49-output problem, plus 11.4-11.8 us generateWinogradTilesKernel that
  regenerates filter tiles on every invocation);
* replacement: one direct fp32 convolution kernel decomposed into 16
  output-channel tiles x 8 input-channel chunks (128 blocks x 512 threads)
  with per-thread partial dots combined by fp32 atomicAdd into the
  zero-initialized output, preceded by one cudaMemsetAsync. No per-call
  weight transform, no im2col buffer, no cuBLAS/cudnn plan lookups.

Previous attempt (trial-b13b4205bb, im2col + strided-batched SGEMM split-K)
was rejected: every split_k choice was much slower end-to-end than the parent
(3.90-5.26 ms vs 1.72-1.79 ms). This candidate therefore uses no torch-op
composition at all: one pybind call, one memset, one custom kernel.

The extension is compiled once during build() (nvcc, cached in the torch
extensions directory), then qualified on the bound device against
F.conv2d on random data (numerical gate + speed gate). If compilation,
qualification or any later guard fails, the callable falls back to the parent
aten.conv2d reference with unchanged semantics.

No CUDA work happens at import time. Model parameters are never transformed,
written or re-packed; weights are read in place. Outputs are freshly allocated
NCHW-contiguous fp32 tensors, matching the parent operator's output layout.
Unobserved shapes/strides/dtypes/scalars fall back to the parent operator.
"""

from __future__ import annotations

import hashlib
import os

import torch

_CALL_IDS = (
    "417724ae1ac74a4e5cbb794e",
    "c74ed56559b58489b279280e",
    "f9b932ed834da4f5ada556b6",
)
_IN_SHAPE = (1, 512, 7, 7)
_W_SHAPE = (512, 512, 3, 3)

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Specialized direct 3x3 convolution for the frozen ResNet-18 stage-4 sites:
// x: [1,512,7,7] fp32 contiguous NCHW; w: [512,512,3,3] fp32 contiguous;
// stride 1, padding 1, dilation 1, groups 1, no bias; out: [1,512,7,7] fp32.
// Decomposition: 16 output-channel tiles of 32 x 8 input-channel chunks of 64
// (grid = 128 blocks of 512 threads); per-thread partial dots are combined by
// fp32 atomicAdd into the zero-initialized output.
__global__ void __launch_bounds__(512) conv3x3_512_7x7_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out) {
  const int ctile = blockIdx.x;   // 0..15
  const int kchunk = blockIdx.y;  // 0..7
  const int c0 = kchunk * 64;
  const int base = ctile * (32 * 49);
  const int end = base + 32 * 49;
  for (int o = base + threadIdx.x; o < end; o += 512) {
    const int p = o % 49;
    const int i = p / 7;
    const int j = p - i * 7;
    const int cout = o / 49;
    const float* wc = w + ((size_t)cout * 512 + (size_t)c0) * 9;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f;
    for (int c = 0; c < 64; ++c, wc += 9) {
      const float* xc = x + (size_t)(c0 + c) * 49;
      float t0 = 0.f, t1 = 0.f, t2 = 0.f;
      float t3 = 0.f, t4 = 0.f, t5 = 0.f;
      float t6 = 0.f, t7 = 0.f, t8 = 0.f;
      if (i > 0) {
        const float* xr = xc + (i - 1) * 7 + j;
        t0 = (j > 0) ? xr[-1] : 0.f;
        t1 = xr[0];
        t2 = (j < 6) ? xr[1] : 0.f;
      }
      {
        const float* xr = xc + i * 7 + j;
        t3 = (j > 0) ? xr[-1] : 0.f;
        t4 = xr[0];
        t5 = (j < 6) ? xr[1] : 0.f;
      }
      if (i < 6) {
        const float* xr = xc + (i + 1) * 7 + j;
        t6 = (j > 0) ? xr[-1] : 0.f;
        t7 = xr[0];
        t8 = (j < 6) ? xr[1] : 0.f;
      }
      a0 += wc[0] * t0 + wc[1] * t1 + wc[2] * t2;
      a1 += wc[3] * t3 + wc[4] * t4 + wc[5] * t5;
      a2 += wc[6] * t6 + wc[7] * t7 + wc[8] * t8;
    }
    atomicAdd(out + o, a0 + a1 + a2);
  }
}

torch::Tensor conv3x3_512_7x7(torch::Tensor x, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat, "fp32 required");
  TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && x.size(1) == 512 && x.size(2) == 7 && x.size(3) == 7,
              "expected x [1,512,7,7]");
  TORCH_CHECK(w.dim() == 4 && w.size(0) == 512 && w.size(1) == 512 && w.size(2) == 3 && w.size(3) == 3,
              "expected w [512,512,3,3]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "contiguous tensors required");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto out = torch::empty({1, 512, 7, 7}, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float* outp = out.data_ptr<float>();
  cudaMemsetAsync(outp, 0, static_cast<size_t>(512) * 49 * sizeof(float), stream);
  dim3 grid(16, 8);
  dim3 block(512);
  conv3x3_512_7x7_kernel<<<grid, block, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), outp);
  return out;
}
"""

_CPP_SOURCE = "torch::Tensor conv3x3_512_7x7(torch::Tensor x, torch::Tensor w);"


def _load_extension():
    if not torch.cuda.is_available():
        # nvcc cannot auto-detect the target arch without a visible device;
        # the bound worker GPU is the leased RTX 4090 (sm_89).
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    from torch.utils.cpp_extension import load_inline

    name = "opop_resnet_s4c3x3_" + hashlib.sha1(_CUDA_SOURCE.encode()).hexdigest()[:10]
    module = load_inline(
        name=name,
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[_CUDA_SOURCE],
        functions=["conv3x3_512_7x7"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return module.conv3x3_512_7x7


def _qualify(fast, device):
    """Setup-time gate on the bound device: enable the fast path only if it is
    numerically faithful against F.conv2d on random data and actually faster
    on this GPU. Any failure disables the fast path (pure parent fallback)."""
    try:
        gen = torch.Generator(device="cpu").manual_seed(20261021)
        x = torch.randn(_IN_SHAPE, generator=gen).to(device)
        w = (torch.randn(_W_SHAPE, generator=gen) * 0.03).to(device)
        ref = torch.nn.functional.conv2d(x, w, None, (1, 1), (1, 1))
        out = fast(x, w)
        if out.shape != ref.shape or out.dtype != ref.dtype or not out.is_contiguous():
            return False
        # Indexing/tiling bugs produce O(1) relative errors; the fp32
        # reduction-order difference vs cudnn is ~1e-6 relative. Generous
        # threshold still separates broken from faithful.
        err = (out - ref).abs().max().item()
        if not (err <= 1e-3 + 1e-3 * ref.abs().max().item()):
            return False

        def _median_time(fn, iters=12, warmup=3):
            for _ in range(warmup):
                fn()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(iters):
                start.record()
                fn()
                stop.record()
                stop.synchronize()
                times.append(start.elapsed_time(stop))
            times.sort()
            return times[len(times) // 2]

        t_fast = _median_time(lambda: fast(x, w))
        t_ref = _median_time(lambda: torch.nn.functional.conv2d(x, w, None, (1, 1), (1, 1)))
        torch.cuda.synchronize(device)
        return t_fast < t_ref
    except Exception:
        return False


def _make_replacement(fast, parent):
    def conv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        if (
            fast is not None
            and bias is None
            and groups == 1
            and isinstance(input, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and input.dtype is torch.float32
            and weight.dtype is torch.float32
            and input.is_cuda
            and weight.is_cuda
            and input.shape == _IN_SHAPE
            and weight.shape == _W_SHAPE
            and input.is_contiguous()
            and weight.is_contiguous()
            and tuple(stride) == (1, 1)
            and tuple(padding) == (1, 1)
            and tuple(dilation) == (1, 1)
        ):
            try:
                return fast(input, weight)
            except Exception:
                pass
        return parent(input, weight, bias, stride, padding, dilation, groups)

    return conv2d


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}

    fast = None
    device = state.get("device")
    try:
        if device is not None and getattr(device, "type", "") == "cuda":
            candidate_fn = _load_extension()
            if _qualify(candidate_fn, device):
                fast = candidate_fn
    except Exception:
        fast = None

    replacements = {}
    for call_id in _CALL_IDS:
        parent = references.get(call_id) or torch.ops.aten.conv2d.default
        replacements[call_id] = _make_replacement(fast, parent)
    return replacements
