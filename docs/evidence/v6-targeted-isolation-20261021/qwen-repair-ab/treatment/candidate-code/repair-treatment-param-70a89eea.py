"""Bit-exact fused RMSNorm repair for the qwen-ttft selection (joint groups
ef1a650b08785209a023 input/post/final norms, 92f04768733eb2543f5d q_norm,
4a79f0c7bb426e51060b k_norm; all prefill phase).

What the previous feedback proves and why the fused-native route is dead
------------------------------------------------------------------------
inputs/previous-feedback.txt rejects trial-recovered-19ccfe875f on QUALITY:
its aten::_fused_rms_norm prefill sites made output.logits fail
atol=rtol=0.02 (max abs 4.09375, 38.8% mismatched), i.e. greedy tokens
diverged. The developer counterfactual ("intervention": unweighted native
fused output, then BF16 weight multiplication - exactly the transformers
rounding order) still failed identically (max abs 2.625, 24.5%), and the
shadow evidence shows that variant differs from the parent at only
192/95M elements by a single bf16 ulp (max 0.0625). Conclusion: any
per-site numeric change that is not bit-exact can flip a near-tie greedy
argmax and fail require_exact_greedy_tokens. A single fused kernel cannot
reproduce aten::mean.dim's accumulation order, so native fusion of the
reduction is unfixable on this task.

What this candidate does instead (bit-exact by construction)
-------------------------------------------------------------
Replace each selected prefill Qwen3RMSNorm.forward with a 3-kernel
composite that keeps the parent's arithmetic bit-for-bit while removing
1015 of the 1160 aten dispatches and most of the fp32 intermediate
traffic (measured family total 2657.7 us inclusive CUDA: to.dtype 669.8,
pow 308.2, mean 561.6, add 144.9, rsqrt 151.3, mul 822.0 across 145
sites, 8 kernels per site):

* Kernel A (custom CUDA): t2 = float(x) * float(x) in one pass over the
  bf16 input. Bitwise equal to the parent's x.float().pow(2) because
  bf16->fp32 is lossless, a bf16-valued fp32 square is exactly
  representable in fp32 (<=16 significand bits), and ATen's
  pow_tensor_scalar special-cases exp==2.0 as base*base (verified on the
  worker's torch 2.13.0: torch.pow(t, 2) equals t*t bitwise on 16M
  random fp32 values including 8M bf16-valued ones).
* aten::mean.dim VERBATIM: torch.mean(t2_view, dim=-1, keepdim=True) on a
  contiguous fp32 tensor with identical shape/strides/values dispatches
  the identical at::native::reduce_kernel<512,1,ReduceOp<float,
  MeanOps<float,float,float,float>,...>> (observed launch configuration)
  and therefore reproduces the parent's reduction bits exactly.
* Kernel B (custom CUDA): out = bf16( float(w) * bf16( float(x) *
  rsqrtf(mean + eps) ) ) in one pass. Each step is the same instruction
  aten's eager chain uses: __fadd_rn == aten add.Tensor(alpha=1) with the
  same double->float scalar conversion, rsqrtf == aten ::rsqrt(float) on
  CUDA (same libdevice function; neither build uses fast-math),
  __fmul_rn == MulFunctor<float> in fp32, __floats2bfloat162_rn == aten
  to.dtype's round-to-nearest-even conversion, and
  float(w)*float(mb)->bf16 == aten's bf16 mul through the float opmath
  path (the observed mul kernel is literally
  BinaryFunctor<BFloat16,...,MulFunctor<float>>). No FMA contraction can
  merge rounding steps: every rounding point is separated by an explicit
  bf16 conversion.

Safety design (mirrors the accepted bundle's defensive pattern):
* One-shot per-site fidelity gate at the first live non-capturing
  invocation: run the composite AND the parent on the live tensor and
  require torch.equal (bit-exact bf16 outputs). Match -> permanent fast
  path; mismatch/exception -> permanent parent fallback with a concise
  route label. Because every op is deterministic and elementwise (no
  atomics) and mean is literally the same aten call, bit-exactness on the
  first live input implies bit-exactness on every later input.
* Stream-capture aware: while torch.cuda.is_current_stream_capturing()
  the call takes the parent path and defers the gate (the gate must
  synchronize), so the required graph bundle trial-fc57bca7b9's capture
  attempt is never disturbed; its eager reference path runs this
  composite bit-exactly, so its own torch.equal verification is
  unaffected. If a capture ever succeeds after gating, the raw kernels
  record like any eager kernel.
* Decode is untouched: only the 145 prefill-phase module sites are
  replaced; decode-phase calls keep the parent path.
* Setup failure category: if the CUDA extension cannot be built on the
  worker, every site routes to "parent-fallback:setup-failure:no-extension"
  instead of silently becoming the parent.
* The workload executes on current inputs every invocation; nothing
  fixture-dependent is cached, no parameter is mutated, and the only
  retained GPU allocation is one shared fp32 scratch buffer (<= 4 MiB,
  sized to the largest site) reused stream-ordered by the t2 stage.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading

import torch

_NORM_SEMANTICS_MARKER = "RMSNorm"
_VECT = 8  # kernel vector width in elements (bf16); guards require numel % 8 == 0

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace {

// Kernel A: t2 = float(x) * float(x)  (bitwise == pow(x.float(), 2)).
// bf16 -> fp32 is lossless and a bf16-valued fp32 square has <= 16
// significand bits, so the single RNE multiply is exact; ATen's
// pow_tensor_scalar computes base*base for exp == 2.0. A lone multiply
// cannot be FMA-contracted.
__global__ void rms_sq_kernel(const __nv_bfloat16* __restrict__ x,
                              float* __restrict__ t2,
                              const long long n_vec) {
  const long long stride = (long long)gridDim.x * (long long)blockDim.x;
  for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n_vec; i += stride) {
    const uint4 p = reinterpret_cast<const uint4*>(x)[i];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&p);
    const float2 f0 = __bfloat1622float2(h[0]);
    const float2 f1 = __bfloat1622float2(h[1]);
    const float2 f2 = __bfloat1622float2(h[2]);
    const float2 f3 = __bfloat1622float2(h[3]);
    float4 lo;
    float4 hi;
    lo.x = f0.x * f0.x;
    lo.y = f0.y * f0.y;
    lo.z = f1.x * f1.x;
    lo.w = f1.y * f1.y;
    hi.x = f2.x * f2.x;
    hi.y = f2.y * f2.y;
    hi.z = f3.x * f3.x;
    hi.w = f3.y * f3.y;
    reinterpret_cast<float4*>(t2)[2 * i] = lo;
    reinterpret_cast<float4*>(t2)[2 * i + 1] = hi;
  }
}

// Kernel B: y = bf16( float(w) * bf16( float(x) * rsqrtf(mean + eps) ) ).
// Every rounding point matches the eager aten chain (module docstring):
// explicit __fadd_rn / __fmul_rn intrinsics, libdevice rsqrtf (aten
// ::rsqrt(float)), and round-to-nearest-even bf16 conversions separating
// each multiply, so no contraction can merge rounding steps.
__global__ void rms_apply_kernel(const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y,
                                 const float* __restrict__ mean,
                                 const __nv_bfloat16* __restrict__ w,
                                 const float eps,
                                 const long long n_vec,
                                 const int n) {
  const long long stride = (long long)gridDim.x * (long long)blockDim.x;
  for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < n_vec; i += stride) {
    const long long base = i * 8;           // first element of this vector
    const long long row = base / n;         // one row per vector (n % 8 == 0)
    const int col = (int)(base - row * n);  // multiple of 8
    const float inv = rsqrtf(__fadd_rn(mean[row], eps));
    const uint4 xp = reinterpret_cast<const uint4*>(x)[i];
    const uint4 wp = reinterpret_cast<const uint4*>(w)[col >> 3];
    const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xp);
    const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wp);
    union Pack {
      __nv_bfloat162 h[4];
      uint4 v;
    } o;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 xf = __bfloat1622float2(xh[k]);
      const float2 wf = __bfloat1622float2(wh[k]);
      const __nv_bfloat162 mb =
          __floats2bfloat162_rn(__fmul_rn(xf.x, inv), __fmul_rn(xf.y, inv));
      const float2 mbf = __bfloat1622float2(mb);
      o.h[k] =
          __floats2bfloat162_rn(__fmul_rn(wf.x, mbf.x), __fmul_rn(wf.y, mbf.y));
    }
    reinterpret_cast<uint4*>(y)[i] = o.v;
  }
}

}  // namespace

void rms_sq(torch::Tensor x, torch::Tensor t2) {
  TORCH_CHECK(x.is_cuda() && t2.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
  TORCH_CHECK(t2.scalar_type() == at::kFloat, "t2 must be fp32");
  TORCH_CHECK(x.is_contiguous() && t2.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(x.numel() % 8 == 0, "numel must be divisible by 8");
  TORCH_CHECK(t2.numel() >= x.numel(), "scratch too small");
  const long long n_vec = x.numel() >> 3;
  if (n_vec == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const long long want = (n_vec + threads - 1) / threads;
  const int blocks = (int)std::min<long long>(want, 32768);
  rms_sq_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      t2.data_ptr<float>(),
      n_vec);
}

void rms_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor w,
               double eps, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && mean.is_cuda() && w.is_cuda() && out.is_cuda(),
              "tensors must be CUDA");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  w.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16,
              "x/w/out must be bf16");
  TORCH_CHECK(mean.scalar_type() == at::kFloat, "mean must be fp32");
  TORCH_CHECK(x.is_contiguous() && mean.is_contiguous() && w.is_contiguous() &&
                  out.is_contiguous(),
              "tensors must be contiguous");
  const long long n_total = x.numel();
  const long long n_last = w.numel();
  TORCH_CHECK(n_last % 8 == 0 && n_total % 8 == 0, "sizes must be divisible by 8");
  TORCH_CHECK(n_total == mean.numel() * n_last, "mean rows mismatch");
  TORCH_CHECK(out.numel() == n_total, "out size mismatch");
  const long long n_vec = n_total >> 3;
  if (n_vec == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const long long want = (n_vec + threads - 1) / threads;
  const int blocks = (int)std::min<long long>(want, 32768);
  rms_apply_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      mean.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      static_cast<float>(eps),
      n_vec,
      (int)n_last);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void rms_sq(torch::Tensor x, torch::Tensor t2);
void rms_apply(torch::Tensor x, torch::Tensor mean, torch::Tensor w, double eps, torch::Tensor out);
"""

_SRC_HASH = hashlib.sha256((_CPP_SRC + _CUDA_SRC).encode()).hexdigest()[:16]

_EXT = {"module": None, "state": "unbuilt"}
_EXT_LOCK = threading.Lock()
_SCRATCH = {"t2": None}


def _ensure_extension():
    """Build (or reuse the cached build of) the CUDA extension exactly once.

    Compilation happens at build()/installation time, never inside the
    measured invocation. A stable build directory keyed by the source hash
    lets later processes (parameter trials) reuse the ninja cache.
    """
    if _EXT["state"] != "unbuilt":
        return
    with _EXT_LOCK:
        if _EXT["state"] != "unbuilt":
            return
        try:
            from torch.utils.cpp_extension import load_inline

            build_directory = os.path.join(
                tempfile.gettempdir(), f"opop-rmsnorm-bitexact-{_SRC_HASH}"
            )
            os.makedirs(build_directory, exist_ok=True)
            module = load_inline(
                name=f"rmsnorm_bitexact_{_SRC_HASH}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["rms_sq", "rms_apply"],
                verbose=False,
                build_directory=build_directory,
            )
            _EXT["module"] = module
            _EXT["state"] = "ready"
        except Exception:
            _EXT["state"] = "failed"


def _no_route(*_args, **_kwargs):
    return None


def _scratch_for(numel, device):
    """One shared fp32 t2 buffer, grown on demand, reused stream-ordered.

    t2 is dead after the aten::mean.dim call inside the same replacement
    invocation, and all sites run sequentially on the current stream, so a
    single buffer serves every site without alias leakage.
    """
    buffer = _SCRATCH.get("t2")
    if buffer is None or buffer.numel() < numel or buffer.device != device:
        buffer = torch.empty(numel, dtype=torch.float32, device=device)
        _SCRATCH["t2"] = buffer
    return buffer


def _compatible(hidden_states, weight):
    return (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.dim() >= 1
        and hidden_states.shape[-1] == weight.shape[-1]
        and hidden_states.numel() > 0
        and hidden_states.numel() % _VECT == 0
        and weight.numel() % _VECT == 0
        and hidden_states.is_contiguous()
        and weight.device == hidden_states.device
    )


def _make_replacement(call_id, module, parent, record_route):
    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 1
        or weight.numel() == 0
        or weight.numel() % _VECT != 0
        or weight.dtype is not torch.bfloat16
        or not weight.is_contiguous()
        or not isinstance(eps, float)
    ):
        def unsupported_module(*args, **kwargs):
            record_route(call_id, "parent-fallback:unsupported-module")
            return parent(*args, **kwargs)

        return unsupported_module

    ext = _EXT["module"]
    if ext is None:

        def setup_failed(*args, **kwargs):
            record_route(call_id, "parent-fallback:setup-failure:no-extension")
            return parent(*args, **kwargs)

        return setup_failed

    def _fast(hidden_states):
        numel = hidden_states.numel()
        t2 = _scratch_for(numel, hidden_states.device)
        ext.rms_sq(hidden_states, t2)
        mean = torch.mean(t2[:numel].view(hidden_states.shape), dim=-1, keepdim=True)
        out = torch.empty_like(hidden_states)
        ext.rms_apply(hidden_states, mean, weight, eps, out)
        return out

    state = {"mode": "unresolved"}

    def replacement(*args, **kwargs):
        mode = state["mode"]
        if len(args) != 1 or kwargs or not _compatible(args[0], weight):
            record_route(call_id, "parent-fallback:unsupported-signature")
            return parent(*args, **kwargs)
        hidden_states = args[0]
        if mode == "fused":
            record_route(call_id, "bitexact-composite")
            return _fast(hidden_states)
        if mode != "unresolved":
            # Permanent parent fallback: fidelity mismatch or gate error.
            record_route(call_id, "parent-fallback:" + mode)
            return parent(hidden_states)
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = True  # capture status unknowable; take the parent path
        if capturing:
            # The gate must synchronize; defer it to a non-capturing call and
            # keep any active capture (required graph bundle) on the parent
            # path.
            record_route(call_id, "parent-fallback:stream-capturing")
            return parent(hidden_states)
        try:
            fused = _fast(hidden_states)
            reference = parent(hidden_states)
            matched = bool(torch.equal(fused, reference))
        except Exception:
            state["mode"] = "gate-error"
            record_route(call_id, "parent-fallback:gate-error")
            return parent(hidden_states)
        if not matched:
            state["mode"] = "fidelity-mismatch-bitexact"
            record_route(call_id, "parent-fallback:fidelity-mismatch-bitexact")
            return reference
        state["mode"] = "fused"
        record_route(call_id, "bitexact-composite-gated")
        return fused

    return replacement


def build(context):
    _ensure_extension()

    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route") or _no_route

    replacements = {}
    for call_id, spec in targets.items():
        if spec.get("kind") != "module" or spec.get("phase") != "prefill":
            continue
        if _NORM_SEMANTICS_MARKER not in str(spec.get("semantics", "")):
            continue
        parent = references.get(call_id)
        module = modules.get(call_id)
        if parent is None or module is None:
            continue
        replacements[call_id] = _make_replacement(
            call_id, module, parent, record_route
        )
    return replacements
