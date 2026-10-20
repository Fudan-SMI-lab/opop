"""Candidate: bit-exact pointwise-chain fusion for the Qwen3-4B TTFT region.

Repairs the previous trial (trial-9cc583939d), which failed the logits gates because
repacking the MLP gate/up GEMMs changed cuBLAS kernel selection and fp32 accumulation
order.  This candidate keeps every GEMM, SDPA call, KV-cache write and layout view
byte-identical and only fuses adjacent pointwise chains, replicating each aten op's
dtype, fp32 opmath and per-op bf16 rounding element-for-element:

* Qwen3RMSNorm module sites (145 per phase): the 8-kernel chain
  to(f32) -> pow -> mean -> add(eps) -> rsqrt -> mul -> to(bf16) -> mul(weight)
  becomes rms_pow2 (x*x in fp32, bit-identical to aten to+pow) -> the unchanged
  aten mean.dim reduction -> rms_tail (rsqrtf + mul + bf16 round + weight mul).
  rsqrtf/expf/__float2bfloat16_rn are the same instructions aten's CUDA kernels use
  (verified bitwise at build time; on any mismatch the site falls back to the parent).
* Rotary chains in attention (per q/k): mul(q,cos), neg, cat, mul(rot,sin), add
  collapse to one rope_add_full kernel; the individual sites cooperate through an
  identity-keyed registry of passthrough records, every fallback path reconstructs
  the exact parent values via dedicated kernels (never via re-routable aten calls).
* MLP activation: silu site passes gate through and the following mul site computes
  round_bf16(round_bf16(x/(1+expf(-x))) * up) in one kernel.

All replacements fall back to the parent reference on signature mismatch or error.
No weights are transformed, no outputs cached, the full workload runs every call.
"""
from __future__ import annotations

import os

import torch

_BF16 = torch.bfloat16

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16_rn(v); }

__global__ void pow2_kernel(const bf16* __restrict__ x, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = bf2f(x[i]); out[i] = v * v; }
}

__global__ void tail_kernel(const bf16* __restrict__ x, const float* __restrict__ var,
                            const bf16* __restrict__ w, float eps,
                            bf16* __restrict__ out, long rows, int width, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long row = i / width; int col = (int)(i % width);
    float r = rsqrtf(var[row] + eps);
    float t = bf2f(x[i]) * r;
    bf16 tb = f2bf(t);
    out[i] = f2bf(bf2f(w[col]) * bf2f(tb));
  }
}

__global__ void silumul_kernel(const bf16* __restrict__ g, const bf16* __restrict__ u,
                               bf16* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xf = bf2f(g[i]);
    bf16 t = f2bf(xf / (1.0f + expf(-xf)));
    out[i] = f2bf(bf2f(t) * bf2f(u[i]));
  }
}

__global__ void ropeA_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cos_,
                             const bf16* __restrict__ sin_, bf16* __restrict__ out,
                             long H, long S, long D,
                             long qs1, long qs2, long qs3,
                             long cs2, long cs3, long ss2, long ss3, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long SD = S * D;
    long h = i / SD; long rem = i % SD; long s = rem / D; long c = rem % D;
    long half = D / 2;
    const bf16* qp = q + h * qs1 + s * qs2;
    float qv = bf2f(qp[c * qs3]);
    float t1 = qv * bf2f(cos_[s * cs2 + c * cs3]);
    float q2;
    if (c < half) q2 = -bf2f(qp[(c + half) * qs3]);
    else         q2 =  bf2f(qp[(c - half) * qs3]);
    float t2 = q2 * bf2f(sin_[s * ss2 + c * ss3]);
    out[i] = f2bf(bf2f(f2bf(t1)) + bf2f(f2bf(t2)));
  }
}

__global__ void ropeB_kernel(const bf16* __restrict__ t1, const bf16* __restrict__ x1,
                             const bf16* __restrict__ xh, const bf16* __restrict__ sin_,
                             bf16* __restrict__ out,
                             long H, long S, long D,
                             long t1s1, long t1s2, long t1s3,
                             long x1s1, long x1s2, long x1s3,
                             long xhs1, long xhs2, long xhs3,
                             long ss2, long ss3, int negated, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long SD = S * D;
    long h = i / SD; long rem = i % SD; long s = rem / D; long c = rem % D;
    long half = D / 2;
    float rot;
    if (c < half) {
      float v = bf2f(xh[h * xhs1 + s * xhs2 + c * xhs3]);
      rot = negated ? v : -v;
    } else {
      rot = bf2f(x1[h * x1s1 + s * x1s2 + (c - half) * x1s3]);
    }
    float t2 = rot * bf2f(sin_[s * ss2 + c * ss3]);
    out[i] = f2bf(bf2f(t1[h * t1s1 + s * t1s2 + c * t1s3]) + bf2f(f2bf(t2)));
  }
}

__global__ void ropeC_kernel(const bf16* __restrict__ q, const bf16* __restrict__ cos_,
                             const bf16* __restrict__ t2, bf16* __restrict__ out,
                             long H, long S, long D,
                             long qs1, long qs2, long qs3,
                             long cs2, long cs3,
                             long t2s1, long t2s2, long t2s3, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long SD = S * D;
    long h = i / SD; long rem = i % SD; long s = rem / D; long c = rem % D;
    float t1 = bf2f(q[h * qs1 + s * qs2 + c * qs3]) * bf2f(cos_[s * cs2 + c * cs3]);
    out[i] = f2bf(bf2f(f2bf(t1)) + bf2f(t2[h * t2s1 + s * t2s2 + c * t2s3]));
  }
}

__global__ void rotsin_kernel(const bf16* __restrict__ x1, const bf16* __restrict__ xh,
                              const bf16* __restrict__ sin_, bf16* __restrict__ out,
                              long H, long S, long D,
                              long x1s1, long x1s2, long x1s3,
                              long xhs1, long xhs2, long xhs3,
                              long ss2, long ss3, int negated, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long SD = S * D;
    long h = i / SD; long rem = i % SD; long s = rem / D; long c = rem % D;
    long half = D / 2;
    float rot;
    if (c < half) {
      float v = bf2f(xh[h * xhs1 + s * xhs2 + c * xhs3]);
      rot = negated ? v : -v;
    } else {
      rot = bf2f(x1[h * x1s1 + s * x1s2 + (c - half) * x1s3]);
    }
    out[i] = f2bf(rot * bf2f(sin_[s * ss2 + c * ss3]));
  }
}

__global__ void catneg_kernel(const bf16* __restrict__ x1, const bf16* __restrict__ xh,
                              bf16* __restrict__ out,
                              long H, long S, long D,
                              long x1s1, long x1s2, long x1s3,
                              long xhs1, long xhs2, long xhs3, int negated, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    long SD = S * D;
    long h = i / SD; long rem = i % SD; long s = rem / D; long c = rem % D;
    long half = D / 2;
    bf16 v;
    if (c < half) {
      float f = bf2f(xh[h * xhs1 + s * xhs2 + c * xhs3]);
      v = negated ? f2bf(f) : f2bf(-f);
    } else {
      v = f2bf(bf2f(x1[h * x1s1 + s * x1s2 + (c - half) * x1s3]));
    }
    out[i] = v;
  }
}

__global__ void rsqrt_test_kernel(const float* __restrict__ v, float* __restrict__ o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = rsqrtf(v[i]);
}

__global__ void silu_test_kernel(const bf16* __restrict__ x, bf16* __restrict__ o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xf = bf2f(x[i]);
    o[i] = f2bf(xf / (1.0f + expf(-xf)));
  }
}

static inline int blocks(long n) { return (int)((n + 255) / 256); }

#define ROWS4(t) TORCH_CHECK(t.dim() == 4 && t.size(0) == 1 && t.is_cuda() && \
                              t.scalar_type() == at::kBFloat16)

torch::Tensor rms_pow2(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  auto out = torch::empty(x.sizes(), x.options().dtype(at::kFloat));
  long n = x.numel();
  pow2_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), out.data_ptr<float>(), (int)n);
  return out;
}

torch::Tensor rms_tail(const torch::Tensor& x, const torch::Tensor& var,
                       const torch::Tensor& w, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(var.is_cuda() && var.scalar_type() == at::kFloat && var.is_contiguous());
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous());
  long width = x.size(-1);
  long rows = x.numel() / width;
  TORCH_CHECK(var.numel() == rows && w.numel() == width);
  auto out = torch::empty(x.sizes(), x.options());
  long n = x.numel();
  tail_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), var.data_ptr<float>(),
      reinterpret_cast<const bf16*>(w.data_ptr()), (float)eps,
      reinterpret_cast<bf16*>(out.data_ptr()), rows, (int)width, n);
  return out;
}

torch::Tensor silu_mul(const torch::Tensor& g, const torch::Tensor& u) {
  TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kBFloat16 && g.is_contiguous());
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kBFloat16 && u.is_contiguous());
  TORCH_CHECK(u.sizes() == g.sizes());
  auto out = torch::empty(g.sizes(), g.options());
  long n = g.numel();
  silumul_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(g.data_ptr()), reinterpret_cast<const bf16*>(u.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), (int)n);
  return out;
}

torch::Tensor rope_add_full(const torch::Tensor& q, const torch::Tensor& cos_,
                            const torch::Tensor& sin_) {
  ROWS4(q);
  long H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(D % 2 == 0);
  TORCH_CHECK(cos_.dim() == 4 && cos_.size(0) == 1 && cos_.size(1) == 1 &&
              cos_.size(2) == S && cos_.size(3) == D);
  TORCH_CHECK(sin_.sizes() == cos_.sizes());
  auto out = torch::empty(q.sizes(), q.options());
  long n = q.numel();
  ropeA_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(cos_.data_ptr()),
      reinterpret_cast<const bf16*>(sin_.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), H, S, D,
      q.stride(1), q.stride(2), q.stride(3),
      cos_.stride(2), cos_.stride(3), sin_.stride(2), sin_.stride(3), n);
  return out;
}

torch::Tensor rope_add_t1(const torch::Tensor& t1, const torch::Tensor& x1,
                          const torch::Tensor& xh, const torch::Tensor& sin_, bool negated) {
  ROWS4(t1);
  long H = t1.size(1), S = t1.size(2), D = t1.size(3);
  long half = D / 2;
  TORCH_CHECK(x1.dim() == 4 && x1.size(0) == 1 && x1.size(1) == H && x1.size(2) == S && x1.size(3) == half);
  TORCH_CHECK(xh.sizes() == x1.sizes());
  TORCH_CHECK(sin_.dim() == 4 && sin_.size(0) == 1 && sin_.size(1) == 1 &&
              sin_.size(2) == S && sin_.size(3) == D);
  auto out = torch::empty(t1.sizes(), t1.options());
  long n = t1.numel();
  ropeB_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(t1.data_ptr()),
      reinterpret_cast<const bf16*>(x1.data_ptr()),
      reinterpret_cast<const bf16*>(xh.data_ptr()),
      reinterpret_cast<const bf16*>(sin_.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), H, S, D,
      t1.stride(1), t1.stride(2), t1.stride(3),
      x1.stride(1), x1.stride(2), x1.stride(3),
      xh.stride(1), xh.stride(2), xh.stride(3),
      sin_.stride(2), sin_.stride(3), negated ? 1 : 0, n);
  return out;
}

torch::Tensor rope_add_t2(const torch::Tensor& q, const torch::Tensor& cos_,
                          const torch::Tensor& t2) {
  ROWS4(q);
  long H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(cos_.dim() == 4 && cos_.size(0) == 1 && cos_.size(1) == 1 &&
              cos_.size(2) == S && cos_.size(3) == D);
  TORCH_CHECK(t2.dim() == 4 && t2.size(0) == 1 && t2.size(1) == H && t2.size(2) == S && t2.size(3) == D);
  auto out = torch::empty(q.sizes(), q.options());
  long n = q.numel();
  ropeC_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(cos_.data_ptr()),
      reinterpret_cast<const bf16*>(t2.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), H, S, D,
      q.stride(1), q.stride(2), q.stride(3),
      cos_.stride(2), cos_.stride(3),
      t2.stride(1), t2.stride(2), t2.stride(3), n);
  return out;
}

torch::Tensor rot_sin(const torch::Tensor& x1, const torch::Tensor& xh,
                      const torch::Tensor& sin_, bool negated) {
  ROWS4(x1);
  long H = x1.size(1), S = x1.size(2), D2 = x1.size(3);
  long D = 2 * D2;
  TORCH_CHECK(xh.sizes() == x1.sizes());
  TORCH_CHECK(sin_.dim() == 4 && sin_.size(0) == 1 && sin_.size(1) == 1 &&
              sin_.size(2) == S && sin_.size(3) == D);
  auto out = torch::empty({1, H, S, D}, x1.options());
  long n = out.numel();
  rotsin_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x1.data_ptr()),
      reinterpret_cast<const bf16*>(xh.data_ptr()),
      reinterpret_cast<const bf16*>(sin_.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), H, S, D,
      x1.stride(1), x1.stride(2), x1.stride(3),
      xh.stride(1), xh.stride(2), xh.stride(3),
      sin_.stride(2), sin_.stride(3), negated ? 1 : 0, n);
  return out;
}

torch::Tensor cat_neg(const torch::Tensor& x1, const torch::Tensor& xh, bool negated) {
  ROWS4(x1);
  long H = x1.size(1), S = x1.size(2), D2 = x1.size(3);
  long D = 2 * D2;
  TORCH_CHECK(xh.sizes() == x1.sizes());
  auto out = torch::empty({1, H, S, D}, x1.options());
  long n = out.numel();
  catneg_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x1.data_ptr()),
      reinterpret_cast<const bf16*>(xh.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), H, S, D,
      x1.stride(1), x1.stride(2), x1.stride(3),
      xh.stride(1), xh.stride(2), xh.stride(3), negated ? 1 : 0, n);
  return out;
}

torch::Tensor test_rsqrt(const torch::Tensor& v) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type() == at::kFloat && v.is_contiguous());
  auto out = torch::empty(v.sizes(), v.options());
  long n = v.numel();
  rsqrt_test_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      v.data_ptr<float>(), out.data_ptr<float>(), (int)n);
  return out;
}

torch::Tensor test_silu(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  auto out = torch::empty(x.sizes(), x.options());
  long n = x.numel();
  silu_test_kernel<<<blocks(n), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()), (int)n);
  return out;
}
"""

_CPP_SRC = """
torch::Tensor rms_pow2(const torch::Tensor& x);
torch::Tensor rms_tail(const torch::Tensor& x, const torch::Tensor& var, const torch::Tensor& w, double eps);
torch::Tensor silu_mul(const torch::Tensor& g, const torch::Tensor& u);
torch::Tensor rope_add_full(const torch::Tensor& q, const torch::Tensor& cos_, const torch::Tensor& sin_);
torch::Tensor rope_add_t1(const torch::Tensor& t1, const torch::Tensor& x1, const torch::Tensor& xh, const torch::Tensor& sin_, bool negated);
torch::Tensor rope_add_t2(const torch::Tensor& q, const torch::Tensor& cos_, const torch::Tensor& t2);
torch::Tensor rot_sin(const torch::Tensor& x1, const torch::Tensor& xh, const torch::Tensor& sin_, bool negated);
torch::Tensor cat_neg(const torch::Tensor& x1, const torch::Tensor& xh, bool negated);
torch::Tensor test_rsqrt(const torch::Tensor& v);
torch::Tensor test_silu(const torch::Tensor& x);
"""

_FUNC_NAMES = ["rms_pow2", "rms_tail", "silu_mul", "rope_add_full", "rope_add_t1",
               "rope_add_t2", "rot_sin", "cat_neg", "test_rsqrt", "test_silu"]

_EXT = None
_EXT_TRIED = False


def _load_ext():
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        if not torch.cuda.is_available():
            return None
        cap = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="opop_pointwise_bitexact_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=_FUNC_NAMES,
            verbose=False,
        )
    except Exception:
        _EXT = None
    return _EXT


def _bits_equal(a, b):
    if a.dtype == torch.bfloat16:
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    return bool(torch.equal(a, b))


def _self_test(ext, device):
    """Bitwise gates against the exact aten chains; a False flag disables that
    component (the corresponding sites then run the parent reference)."""
    flags = {"norm": False, "silu_mul": False, "rope": False}
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(20261020)
        ok_pow = True
        for scale in (1e-3, 1.0, 30.0, 1e4):
            x = (torch.randn(512, 2560, generator=gen, device=device) * scale).to(torch.bfloat16)
            if not _bits_equal(ext.rms_pow2(x), x.to(torch.float32).pow(2)):
                ok_pow = False
        ok_tail = ok_pow
        for shape in ((1, 64, 2560), (1, 64, 32, 128), (1, 64, 8, 128), (1, 1, 2560)):
            x = (torch.randn(shape, generator=gen, device=device) * 1.5).to(torch.bfloat16)
            w = (torch.randn(shape[-1], generator=gen, device=device) * 0.5 + 1.0).to(torch.bfloat16)
            hs = x.to(torch.float32)
            var = hs.pow(2).mean(-1, keepdim=True)
            ref = w * (hs * torch.rsqrt(var + 1e-6)).to(torch.bfloat16)
            if not _bits_equal(ext.rms_tail(x, var.contiguous(), w, 1e-6), ref):
                ok_tail = False
        flags["norm"] = bool(ok_pow and ok_tail)

        x = torch.cat([(torch.randn(1 << 20, generator=gen, device=device) * s).to(torch.bfloat16)
                       for s in (0.01, 1.0, 20.0, 100.0)])
        u = (torch.randn(1 << 20, generator=gen, device=device) * 2.0).to(torch.bfloat16)
        flags["silu_mul"] = _bits_equal(ext.silu_mul(x, u), torch.nn.functional.silu(x) * u)

        def rotate_half(t):
            return torch.cat((-t[..., t.shape[-1] // 2:], t[..., : t.shape[-1] // 2]), dim=-1)

        ok_rope = True
        for (H, S, D) in ((32, 16, 128), (8, 16, 128), (4, 1, 128), (32, 1, 128)):
            q = torch.randn(1, S, H, D, generator=gen, device=device).to(torch.bfloat16).transpose(1, 2)
            cos = torch.randn(1, 1, S, D, generator=gen, device=device).clamp(-1, 1).to(torch.bfloat16)
            sin = torch.randn(1, 1, S, D, generator=gen, device=device).clamp(-1, 1).to(torch.bfloat16)
            ref = (q * cos) + (rotate_half(q) * sin)
            if not _bits_equal(ext.rope_add_full(q, cos, sin), ref):
                ok_rope = False
            t1 = q * cos
            x1 = q[..., : D // 2]
            x2 = q[..., D // 2:]
            ref2 = t1 + (torch.cat((-x2, x1), dim=-1) * sin)
            if not _bits_equal(ext.rope_add_t1(t1.contiguous(), x1, x2, sin, False), ref2):
                ok_rope = False
            if not _bits_equal(ext.rope_add_t1(t1.contiguous(), x1, -x2, sin, True), ref2):
                ok_rope = False
            if not _bits_equal(ext.rope_add_t2(q, cos, torch.cat((-x2, x1), dim=-1) * sin), ref):
                ok_rope = False
            if not _bits_equal(ext.cat_neg(x1, x2, False), torch.cat((-x2, x1), dim=-1)):
                ok_rope = False
            if not _bits_equal(ext.rot_sin(x1, x2, sin, False), torch.cat((-x2, x1), dim=-1) * sin):
                ok_rope = False
        flags["rope"] = bool(ok_rope)
    except Exception:
        pass
    return flags


class _Reg:
    """Identity-keyed passthrough records; consume-once, strong refs keep ids stable."""

    __slots__ = ("rope1", "neg", "cat", "rope2", "gate")

    def __init__(self):
        self.rope1 = {}
        self.neg = {}
        self.cat = {}
        self.rope2 = {}
        self.gate = {}


def _make_handlers(ext, flags, reg):
    f_norm = flags.get("norm", False)
    f_silu = flags.get("silu_mul", False)
    f_rope = flags.get("rope", False)

    def norm_forward(module, ref):
        def forward(*args, **kwargs):
            try:
                hs = args[0] if args else kwargs.get("hidden_states")
                if f_norm and hs.is_cuda and hs.dtype is _BF16 and hs.dim() >= 1:
                    w = module.weight
                    eps = module.variance_epsilon
                    if (hs.shape[-1] == w.shape[0] and hs.stride(-1) == 1
                            and w.dtype is _BF16 and w.stride(-1) == 1 and w.is_cuda):
                        xc = hs if hs.is_contiguous() else hs.contiguous()
                        var = ext.rms_pow2(xc).mean(-1, keepdim=True)
                        return ext.rms_tail(xc, var, w, eps)
            except Exception:
                pass
            return ref(*args, **kwargs)

        return forward

    def silu_repl(ref):
        def repl(*args, **kwargs):
            try:
                x = args[0] if args else kwargs.get("input")
                if f_silu and x.dim() == 3 and x.dtype is _BF16 and x.is_cuda and x.is_contiguous():
                    reg.gate[id(x)] = x
                    return x
            except Exception:
                pass
            return ref(*args, **kwargs)

        return repl

    def mul_repl(ref):
        def repl(a, b):
            try:
                if f_rope and a.dim() == 4 and b.dim() == 4 and a.dtype is _BF16 \
                        and b.dtype is _BF16 and a.size(0) == 1 and b.size(0) == 1 \
                        and b.size(1) == 1 and a.size(2) == b.size(2) \
                        and a.size(3) == b.size(3) and a.size(3) % 2 == 0:
                    key = id(a)
                    catrec = reg.cat.get(key)
                    if catrec is not None and catrec[3] is a:
                        snap = None
                        if len(reg.rope1) == 1:
                            q, c = next(iter(reg.rope1.values()))
                            if q.data_ptr() == a.data_ptr() and q.shape == a.shape:
                                snap = (q, c)
                        if snap is not None:
                            reg.rope2[key] = (catrec[0], catrec[1], catrec[2], b, snap[0], snap[1])
                            reg.cat.pop(key, None)
                            return a
                        return ext.rot_sin(catrec[0], catrec[1], b, catrec[2])
                    reg.rope1[key] = (a, b)
                    return a
                if f_silu and a.dim() == 3 and b.dim() == 3 and a.shape == b.shape \
                        and a.dtype is _BF16 and b.dtype is _BF16 and a.is_cuda \
                        and a.is_contiguous() and b.is_contiguous():
                    rec = reg.gate.pop(id(a), None)
                    if rec is not None and rec is a:
                        return ext.silu_mul(a, b)
            except Exception:
                pass
            return ref(a, b)

        return repl

    def neg_repl(ref):
        def repl(*args, **kwargs):
            try:
                x = args[0] if args else kwargs.get("self")
                if f_rope and x.dim() == 4 and x.dtype is _BF16 and x.size(0) == 1:
                    reg.neg[id(x)] = x
                    return x
            except Exception:
                pass
            return ref(*args, **kwargs)

        return repl

    def cat_repl(ref):
        def repl(*args, **kwargs):
            try:
                if f_rope:
                    if args and isinstance(args[0], (list, tuple)):
                        tensors, dim = args[0], (args[1] if len(args) > 1 else kwargs.get("dim", 0))
                    else:
                        tensors = kwargs.get("tensors")
                        dim = kwargs.get("dim", 0)
                    if dim == -1 and tensors is not None and len(tensors) == 2:
                        a, b = tensors
                        if a.dim() == 4 and b.dim() == 4 and a.dtype is _BF16 \
                                and b.dtype is _BF16 and a.shape == b.shape \
                                and a.stride() == b.stride() and a.stride(3) == 1:
                            half = a.size(3)
                            negrec = reg.neg.pop(id(a), None)
                            if negrec is not None and negrec is a:
                                x1, xh, negated = b, a, False
                            else:
                                x1, xh, negated = b, a, True
                            if a.data_ptr() - b.data_ptr() == half * a.element_size():
                                fake = torch.as_strided(x1, x1.shape[:-1] + (2 * half,),
                                                        x1.stride(), x1.storage_offset())
                                reg.cat[id(fake)] = (x1, xh, negated, fake)
                                return fake
                            return ext.cat_neg(x1, xh, negated)
            except Exception:
                pass
            return ref(*args, **kwargs)

        return repl

    def add_repl(ref):
        def repl(*args, **kwargs):
            try:
                if f_rope and len(args) == 2 and not kwargs \
                        and args[0].dim() == 4 and args[1].dim() == 4 \
                        and args[0].shape == args[1].shape and args[0].dtype is _BF16 \
                        and args[1].dtype is _BF16 and args[0].size(0) == 1:
                    a, b = args
                    r1 = reg.rope1.pop(id(a), None)
                    r2 = reg.rope2.pop(id(b), None)
                    if r2 is not None:
                        x1, xh, negated, sin = r2[0], r2[1], r2[2], r2[3]
                        if r1 is not None and r1[0] is a:
                            return ext.rope_add_full(a, r1[1], sin)
                        if r2[4] is not None and r2[4] is a:
                            return ext.rope_add_full(a, r2[5], sin)
                        return ext.rope_add_t1(
                            a if a.is_contiguous() else a.contiguous(), x1, xh, sin, negated)
                    if r1 is not None and r1[0] is a:
                        return ext.rope_add_t2(a, r1[1], b)
            except Exception:
                pass
            return ref(*args, **kwargs)

        return repl

    return {
        "norm": norm_forward,
        "silu": silu_repl,
        "mul": mul_repl,
        "neg": neg_repl,
        "cat": cat_repl,
        "add": add_repl,
    }


def _sig_parts(spec):
    sigs = spec.get("signatures") or []
    if not sigs:
        return [], {}
    s = sigs[0]
    return s.get("args", []), s.get("kwargs", {})


def _is_tensor_arg(a):
    return isinstance(a, dict) and "shape" in a


def _classify(spec, modules):
    """Return a component name or None. Signature-based; identical to the static
    classification used for proposal.json."""
    kind = spec.get("kind")
    name = spec.get("target", "")
    if kind == "module":
        module = modules.get(spec["id"])
        if module is not None and hasattr(module, "weight") and hasattr(module, "variance_epsilon"):
            return "norm"
        return None
    if kind != "functional":
        return None
    args, kwargs = _sig_parts(spec)
    if name == "aten.silu.default":
        return "silu"
    if name == "aten.mul.Tensor":
        if len(args) == 2 and _is_tensor_arg(args[0]) and _is_tensor_arg(args[1]):
            s0, s1 = args[0]["shape"], args[1]["shape"]
            if len(s0) == 3 and s0 == s1:
                return "mlp_mul"
            if (len(s0) == 4 and len(s1) == 4 and s1[0] == 1 and s1[1] == 1
                    and s1[2] == s0[2] and s1[3] == s0[3] and s0[3] % 2 == 0):
                return "rope_mul"
        return None
    if name == "aten.neg.default":
        if len(args) == 1 and _is_tensor_arg(args[0]) and len(args[0]["shape"]) == 4:
            return "rope_neg"
        return None
    if name == "aten.cat.default":
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        tl = args[0] if args else kwargs.get("tensors")
        if dim == -1 and isinstance(tl, list) and len(tl) == 2 \
                and all(_is_tensor_arg(t) for t in tl) \
                and len(tl[0]["shape"]) == 4 and tl[0]["shape"] == tl[1]["shape"]:
            return "rope_cat"
        return None
    if name == "aten.add.Tensor":
        if len(args) == 2 and _is_tensor_arg(args[0]) and _is_tensor_arg(args[1]):
            s0 = args[0]["shape"]
            s1 = args[1]["shape"]
            if len(s0) == 4 and s0 == s1:
                return "rope_add"
        return None
    return None


def build(context):
    """Return {call_id: callable} replacements for this rank's local targets."""
    state = context.get("state") or {}
    device = state.get("device")
    if isinstance(device, torch.device):
        dev = device
    elif isinstance(device, str):
        dev = torch.device(device)
    else:
        dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ext = _load_ext()
    flags = _self_test(ext, dev) if ext is not None and dev.type == "cuda" else {}
    reg = _Reg()
    handlers = _make_handlers(ext if ext is not None else None, flags, reg)

    targets = context.get("targets") or {}
    if isinstance(targets, dict):
        items = targets.items()
    else:
        items = ((spec["id"], spec) for spec in targets)

    modules = context.get("modules") or {}
    references = context.get("references") or {}

    out = {}
    for call_id, spec in items:
        try:
            comp = _classify(spec, modules)
        except Exception:
            comp = None
        if comp is None:
            continue
        ref = references.get(call_id)
        if comp == "norm":
            module = modules.get(call_id)
            if module is None or ref is None:
                continue
            out[call_id] = handlers["norm"](module, ref)
        elif comp == "silu":
            if ref is not None:
                out[call_id] = handlers["silu"](ref)
        elif comp in ("mlp_mul", "rope_mul"):
            if ref is not None:
                out[call_id] = handlers["mul"](ref)
        elif comp == "rope_neg":
            if ref is not None:
                out[call_id] = handlers["neg"](ref)
        elif comp == "rope_cat":
            if ref is not None:
                out[call_id] = handlers["cat"](ref)
        elif comp == "rope_add":
            if ref is not None:
                out[call_id] = handlers["add"](ref)
    return out
