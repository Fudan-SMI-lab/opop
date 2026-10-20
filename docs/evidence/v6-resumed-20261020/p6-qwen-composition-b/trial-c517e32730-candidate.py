"""Candidate: prefill-only GEMM replacements for the Qwen3-4B TTFT boundary.

Complements (never replaces) the installed pointwise region bundle
trial-resample-3896551a-param-102dba91: the six selected prefill aten::linear
shape groups share no call with the bundle, so both implementations execute.

Repairs addressed from inputs/previous-feedback.txt and prior trials:
* Failed decode CUDA-graph proposals (including an unconditional logits.clone
  variant) and a decode fusion that silently fell back on every call are avoided
  entirely: this candidate touches only prefill aten::linear sites, wraps no
  logits and captures no graphs.
* trial-9cc583939d repacked the MLP gate/up GEMMs, changed cuBLAS kernel
  selection and fp32 accumulation order and broke the 0.02 logits /
  exact-greedy-token gates. This candidate therefore leaves gate/up (72 sites),
  down_proj (36), o_proj (36) and q_proj (36) on the parent aten::linear path
  and replaces only the two measured-inefficient prefill GEMM families:

  1. lm_head prefill matvec (1 call: [1,1,2560] x [151936,2560]^T), observed as
     internal::gemvx streaming 778 MB of weights at ~0.93 TB/s (roughly half of
     achievable Ampere DRAM streaming, 833 us): replaced by a warp-per-row CUDA
     matvec with 16-byte vectorized loads and fp32 accumulation, bf16
     round-to-nearest output. Only the fp32 summation order differs from the
     parent, which lands within ~1 bf16 ULP.
  2. k_proj/v_proj prefill GEMMs (72 calls: [1,256,2560] x [1024,2560]^T),
     observed as a cutlass_80_tensorop splitK kernel plus a cublasLt
     splitKreduce per call (144 launches and workspace round-trips, ~86 TF vs
     ~128 TF sibling shapes): replaced by one merged cuBLAS GEMM per layer over
     an invariant packed [Wk;Wv] (2048x2560) weight built once per physical
     module at build time. The k_proj call computes the merged output and
     returns the k half as a view; the v_proj call returns the stashed v half
     after input/weight identity verification (consume-once registry with
     strong refs, the installed bundle's pattern). Downstream consumers remain
     valid because Qwen3 attention only splits the last projection dimension
     (.view(bsz, q_len, kv_heads, head_dim) is a legal pure split of the slice)
     and every later op is stride-agnostic or falls back inside the installed
     bundle bit-exactly.

Every custom path carries (a) a build-time numeric gate against the parent
operator on synthetic data, (b) a build-time speed gate, and (c) a per-site
first-call live numeric comparison against the parent reference with permanent
per-site fallback. All replacements fall back to the parent reference on any
signature mismatch or error. No outputs are cached and the full workload runs
on every invocation; actual execution is reported through record_route labels
(lmhead_matvec / lmhead_first_call_ok / lmhead_first_call_reject /
lmhead_parent, kv_merge_fused / kv_first_call_ok / kv_first_call_reject /
kv_merge_consume / kv_parent) instead of being assumed.
"""
from __future__ import annotations

import os
import time

import torch

_BF16 = torch.bfloat16
_ATOL = 2.0 ** -7
_RTOL = 2.0 ** -7

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16_rn(v); }

__global__ void lmhead_mv_kernel(const bf16* __restrict__ W, const bf16* __restrict__ x,
                                 bf16* __restrict__ y, int K, long N) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long row = (long)blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= N) return;
  const uint4* wp = reinterpret_cast<const uint4*>(W + row * (long)K);
  const uint4* xp = reinterpret_cast<const uint4*>(x);
  const int chunks = K >> 3;
  float acc = 0.f;
  for (int c = lane; c < chunks; c += 32) {
    const uint4 wv = __ldg(wp + c);
    const uint4 xv = __ldg(xp + c);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __bfloat1622float2(w2[j]);
      const float2 xf = __bfloat1622float2(x2[j]);
      acc = fmaf(wf.x, xf.x, acc);
      acc = fmaf(wf.y, xf.y, acc);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) y[row] = f2bf(acc);
}

torch::Tensor lmhead_matvec(const torch::Tensor& x, const torch::Tensor& W) {
  TORCH_CHECK(x.is_cuda() && W.is_cuda());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && W.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.is_contiguous() && W.is_contiguous());
  const long N = W.size(0);
  const long K = W.size(1);
  TORCH_CHECK(x.numel() == K && (K & 255) == 0);
  auto out = torch::empty({1, 1, N}, W.options());
  const int warps = 8;
  const long blocks = (N + warps - 1) / warps;
  lmhead_mv_kernel<<<(int)blocks, warps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(W.data_ptr()),
      reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<bf16*>(out.data_ptr()), (int)K, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP_SRC = "torch::Tensor lmhead_matvec(const torch::Tensor& x, const torch::Tensor& W);"

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
            name="opop_prefill_gemm_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["lmhead_matvec"],
            verbose=False,
        )
    except Exception:
        _EXT = None
    return _EXT


def _close(a, b):
    af = a.detach().float()
    bf = b.detach().float()
    return bool(torch.all((af - bf).abs() <= _ATOL + _RTOL * bf.abs()))


def _median_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    return samples[len(samples) // 2] / 1e3


def _gate_lmhead(ext, device):
    if ext is None or device.type != "cuda":
        return False
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(20261021)
        w = (torch.randn(151936, 2560, generator=gen, device=device) * 0.03).to(_BF16)
        ok = True
        for scale in (0.25, 1.0, 4.0):
            x = (torch.randn(1, 1, 2560, generator=gen, device=device) * scale).to(_BF16)
            if not _close(ext.lmhead_matvec(x, w), torch.nn.functional.linear(x, w)):
                ok = False
                break
        if not ok:
            return False
        x = torch.randn(1, 1, 2560, generator=gen, device=device).to(_BF16)
        ref_us = _median_us(lambda: torch.nn.functional.linear(x, w))
        new_us = _median_us(lambda: ext.lmhead_matvec(x, w))
        return new_us < 0.85 * ref_us
    except Exception:
        return False


def _gate_kv(device):
    if device.type != "cuda":
        return False
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(20261022)
        x = torch.randn(1, 256, 2560, generator=gen, device=device).to(_BF16)
        wk = (torch.randn(1024, 2560, generator=gen, device=device) * 0.03).to(_BF16)
        wv = (torch.randn(1024, 2560, generator=gen, device=device) * 0.03).to(_BF16)
        wkv = torch.cat([wk, wv], dim=0)
        lin = torch.nn.functional.linear
        merged = lin(x, wkv)
        if not (_close(merged[..., :1024], lin(x, wk)) and _close(merged[..., 1024:], lin(x, wv))):
            return False

        def separate():
            lin(x, wk)
            lin(x, wv)

        ref_us = _median_us(separate)
        new_us = _median_us(lambda: lin(x, wkv))
        return new_us < 0.88 * ref_us
    except Exception:
        return False


def _pack_kv(models):
    model = (models or {}).get("model")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return None
    kreg = {}
    vset = set()
    try:
        with torch.no_grad():
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                wk = getattr(getattr(attn, "k_proj", None), "weight", None)
                wv = getattr(getattr(attn, "v_proj", None), "weight", None)
                if not (isinstance(wk, torch.Tensor) and isinstance(wv, torch.Tensor)):
                    return None
                if wk.shape != (1024, 2560) or wv.shape != (1024, 2560):
                    return None
                if wk.device.type != "cuda" or wv.device.type != "cuda":
                    return None
                if wk.dtype is not _BF16 or wv.dtype is not _BF16:
                    return None
                if not (wk.is_contiguous() and wv.is_contiguous()):
                    return None
                if wk.data_ptr() == wv.data_ptr():
                    return None
                wkv = torch.cat([wk, wv], dim=0)
                if wkv.shape != (2048, 2560) or not wkv.is_contiguous():
                    return None
                kreg[wk.data_ptr()] = {"wkv": wkv, "wv_ptr": wv.data_ptr(), "half": 1024}
                vset.add(wv.data_ptr())
    except Exception:
        return None
    return (kreg, vset) if kreg else None


def _kv_sig_ok(x, w):
    return (x.dim() == 3 and tuple(x.shape) == (1, 256, 2560) and w.dim() == 2
            and tuple(w.shape) == (1024, 2560) and x.dtype is _BF16 and w.dtype is _BF16
            and x.is_cuda and w.is_cuda)


def _lm_repl(call_id, ref, ext, record, enabled):
    state = {"verified": False, "ok": bool(enabled)}

    def repl(*args, **kwargs):
        try:
            if (state["ok"] and ext is not None and not kwargs and len(args) == 2):
                x, w = args
                if (x.dim() == 3 and tuple(x.shape) == (1, 1, 2560) and w.dim() == 2
                        and tuple(w.shape) == (151936, 2560) and x.dtype is _BF16
                        and w.dtype is _BF16 and x.is_cuda and w.is_cuda
                        and x.is_contiguous() and w.is_contiguous()):
                    out = ext.lmhead_matvec(x, w)
                    if not state["verified"]:
                        state["verified"] = True
                        if _close(out, ref(x, w)):
                            record(call_id, "lmhead_first_call_ok")
                        else:
                            state["ok"] = False
                            record(call_id, "lmhead_first_call_reject")
                            return ref(*args, **kwargs)
                    record(call_id, "lmhead_matvec")
                    return out
        except Exception:
            pass
        record(call_id, "lmhead_parent")
        return ref(*args, **kwargs)

    return repl


def _kv_repl(call_id, ref, kreg, vset, stash, record, enabled):
    state = {"verified": False, "ok": bool(enabled)}

    def repl(*args, **kwargs):
        try:
            if args and len(args) >= 2 and not (set(kwargs) - {"bias"}):
                bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
                x, w = args[0], args[1]
                if bias is None and isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor):
                    ptr = w.data_ptr()
                    ent = kreg.get(ptr)
                    if ent is not None and state["ok"] and _kv_sig_ok(x, w):
                        merged = torch.nn.functional.linear(x, ent["wkv"])
                        k = merged[..., : ent["half"]]
                        if not state["verified"]:
                            state["verified"] = True
                            refk = ref(x, w)
                            if _close(k, refk):
                                record(call_id, "kv_first_call_ok")
                            else:
                                state["ok"] = False
                                record(call_id, "kv_first_call_reject")
                                return refk
                        stash[id(x)] = (x, ent["wv_ptr"], merged[..., ent["half"]:])
                        record(call_id, "kv_merge_fused")
                        return k
                    if ptr in vset and _kv_sig_ok(x, w):
                        rec = stash.pop(id(x), None)
                        if rec is not None and rec[0] is x and rec[1] == ptr:
                            record(call_id, "kv_merge_consume")
                            return rec[2]
        except Exception:
            pass
        record(call_id, "kv_parent")
        return ref(*args, **kwargs)

    return repl


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

    record = context.get("record_route") or (lambda *_: None)

    ext = None
    lm_ok = False
    kv_pack = None
    if dev.type == "cuda" and torch.cuda.is_available():
        ext = _load_ext()
        lm_ok = _gate_lmhead(ext, dev)
        if _gate_kv(dev):
            kv_pack = _pack_kv(context.get("models") or {})

    kreg, vset = kv_pack if kv_pack is not None else ({}, set())
    stash = {}

    targets = context.get("targets") or {}
    if isinstance(targets, dict):
        items = targets.items()
    else:
        items = ((spec["id"], spec) for spec in targets)
    references = context.get("references") or {}

    out = {}
    for call_id, spec in items:
        try:
            if spec.get("kind") != "functional" or spec.get("target") != "aten.linear.default":
                continue
            sigs = spec.get("signatures") or []
            if not sigs:
                continue
            sargs = sigs[0].get("args") or []
            if len(sargs) < 2 or not (isinstance(sargs[0], dict) and isinstance(sargs[1], dict)):
                continue
            xshape = tuple(sargs[0].get("shape") or ())
            wshape = tuple(sargs[1].get("shape") or ())
            ref = references.get(call_id)
            if ref is None:
                ref = torch.nn.functional.linear
            if xshape == (1, 1, 2560) and wshape == (151936, 2560):
                out[call_id] = _lm_repl(call_id, ref, ext, record, lm_ok)
            elif xshape == (1, 256, 2560) and wshape == (1024, 2560):
                out[call_id] = _kv_repl(call_id, ref, kreg, vset, stash, record, kv_pack is not None)
        except Exception:
            continue
    return out
