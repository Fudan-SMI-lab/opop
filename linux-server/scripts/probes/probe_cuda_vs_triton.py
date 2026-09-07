"""Step 8: does a hand-written CUDA kernel beat Triton on the SAME algorithm, our timing?

The question is narrow on purpose. The user's position -- which KernelPro supports (it reports
raw-CUDA+CuTe at 1.23x over hand-tuned Triton on a MoE kernel) -- is that CUDA's expressiveness
ceiling is higher than Triton's, so a CUDA candidate should reach further. Our harness generates
Triton by default. Before spending search budget teaching it to generate CUDA, measure whether
the gap is real ON OUR TIMING CONVENTION.

WHAT IS AND IS NOT MEASURED. One hand-written pair is one data point about THESE two
implementations, and a hand-written kernel is only as good as the hand that wrote it. It cannot
establish "CUDA is faster than Triton" in general. What it can do is bound the prize: if a
competently-written CUDA version of the same algorithm cannot beat its Triton twin, the backend
work buys little on this class of kernel.

FAIRNESS RULES, since an unfair comparison here is worse than no measurement:
  1. Same algorithm, same tiling, same precision path (tf32 tensor cores, fp32 accumulate).
     Not "a good CUDA kernel vs a naive Triton one".
  2. BOTH get a config sweep of comparable size. Timing one tuned implementation against one
     untuned one manufactures the conclusion. The CUDA side is templated over the same tiling
     axes Triton exposes (BM/BN/BK, warp arrangement, double-buffering ~ num_stages) so the
     sweep is over the same decisions.
  3. The CUDA side gets the techniques a competent author would use: 128-bit vector loads,
     shared-memory staging, and a ping-pong double buffer standing in for Triton's software
     pipelining. Writing scalar loads and calling the result "CUDA's ceiling" would be a
     measurement of my own laziness.
  4. Both timed by the SAME function, our harness's convention (CUDA events + per-trial L2
     flush), medians over the same trial count.
  5. Correctness FIRST, and it GATES. A wrong kernel's latency is not reported as a data point
     -- a config that fails the check is excluded from the sweep entirely, and if no CUDA
     config is correct the probe fails rather than printing numbers.
"""
from __future__ import annotations

import statistics
import sys

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

DEV = "cuda"
M, N, K = 1024, 1024, 1024
REL_TOL = 1e-2          # tf32 vs tf32: real errors here are O(1), not O(tol)


def flush_l2():
    """Same L2 eviction our harness's timing does between trials."""
    x = torch.empty(int(64 * 1024 * 1024), dtype=torch.int8, device=DEV)
    x.zero_()


def timed(fn, trials=100, warmup=25):
    """CUDA events + per-trial L2 flush: our harness's convention, not a wall-clock loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        flush_l2()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), statistics.mean(out)


# ---------------------------------------------------------------- Triton
@triton.jit
def _tl_matmul(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + offs_m[:, None] * sam + offs_k[None, :] * sak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # tf32 on both sides: the precision path must match the CUDA kernel's, or this
        # measures a precision difference and calls it a backend difference.
        acc += tl.dot(a, b, input_precision="tf32")
    tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_mm(a, b, BM, BN, BK, num_warps, num_stages):
    c = torch.empty((M, N), device=DEV, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _tl_matmul[grid](a, b, c, M, N, K,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1),
                     BM=BM, BN=BN, BK=BK,
                     num_warps=num_warps, num_stages=num_stages)
    return c


TRITON_CONFIGS = [
    (128, 128, 32, 8, 3), (128, 128, 32, 8, 4), (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4), (64, 64, 32, 4, 4), (128, 128, 16, 8, 4),
    (64, 64, 64, 4, 3),
]

# ---------------------------------------------------------------- CUDA
# Templated over the SAME axes Triton exposes: tile shape, warp arrangement (Triton's
# num_warps), and double-buffering (Triton's num_stages). One instantiation per config so
# the sweep is over the same decisions, not over incidental code differences.
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// tf32 wmma shape. m16n16k8 is the tf32 fragment on sm_80+.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, bool DBUF>
__global__ __launch_bounds__(WARPS_M * WARPS_N * 32)
void mm_tf32(const float* __restrict__ A, const float* __restrict__ B,
             float* __restrict__ C, int M, int N, int K) {
  constexpr int NBUF = DBUF ? 2 : 1;
  constexpr int WM = BM / WARPS_M;          // this warp's output tile
  constexpr int WN = BN / WARPS_N;
  constexpr int FM = WM / WMMA_M;           // fragments per warp
  constexpr int FN = WN / WMMA_N;
  constexpr int THREADS = WARPS_M * WARPS_N * 32;

  __shared__ float As[NBUF][BM][BK];
  __shared__ float Bs[NBUF][BK][BN];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  // Warp grid is WARPS_M x WARPS_N and covers the whole tile exactly -- the earlier version
  // mapped 8 warps onto a 2x2 grid of 64x64 warp tiles, so warps 4..7 addressed rows past
  // the tile and the kernel was silently wrong (max rel err 1.26).
  const int warp_m = (warp / WARPS_N) * WM;
  const int warp_n = (warp % WARPS_N) * WN;

  const int block_m = blockIdx.x * BM;
  const int block_n = blockIdx.y * BN;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  // ---- staged loads. float4 on the fast path (K, N and the tile extents are all
  // multiples of 4 and torch pointers are 16B-aligned), scalar+zero-fill on the edges.
  auto load_tile = [&](int k0, int buf) {
    const bool full_a = (block_m + BM <= M) && (k0 + BK <= K);
    if (full_a && (BK % 4 == 0)) {
      for (int i = tid * 4; i < BM * BK; i += THREADS * 4) {
        int r = i / BK, c = i % BK;
        const float4 v = *reinterpret_cast<const float4*>(&A[(long)(block_m + r) * K + k0 + c]);
        *reinterpret_cast<float4*>(&As[buf][r][c]) = v;
      }
    } else {
      for (int i = tid; i < BM * BK; i += THREADS) {
        int r = i / BK, c = i % BK;
        int gr = block_m + r, gc = k0 + c;
        As[buf][r][c] = (gr < M && gc < K) ? A[(long)gr * K + gc] : 0.0f;
      }
    }
    const bool full_b = (block_n + BN <= N) && (k0 + BK <= K);
    if (full_b && (BN % 4 == 0)) {
      for (int i = tid * 4; i < BK * BN; i += THREADS * 4) {
        int r = i / BN, c = i % BN;
        const float4 v = *reinterpret_cast<const float4*>(&B[(long)(k0 + r) * N + block_n + c]);
        *reinterpret_cast<float4*>(&Bs[buf][r][c]) = v;
      }
    } else {
      for (int i = tid; i < BK * BN; i += THREADS) {
        int r = i / BN, c = i % BN;
        int gr = k0 + r, gc = block_n + c;
        Bs[buf][r][c] = (gr < K && gc < N) ? B[(long)gr * N + gc] : 0.0f;
      }
    }
  };

  auto compute_tile = [&](int buf) {
    for (int kk = 0; kk < BK; kk += WMMA_K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major> af[FM];
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                     wmma::precision::tf32, wmma::row_major> bf[FN];
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(af[i], &As[buf][warp_m + i * WMMA_M][kk], BK);
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(bf[j], &Bs[buf][kk][warp_n + j * WMMA_N], BN);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
          wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
  };

  if (DBUF) {
    // Ping-pong: stage the next K-slab while the current one is being consumed. This is the
    // hand-written stand-in for Triton's num_stages software pipeline. Writes go to the OTHER
    // buffer, so one barrier per iteration is sufficient.
    int buf = 0;
    load_tile(0, buf);
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += BK) {
      const int nxt = buf ^ 1;
      if (k0 + BK < K) load_tile(k0 + BK, nxt);
      compute_tile(buf);
      __syncthreads();
      buf = nxt;
    }
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      load_tile(k0, 0);
      __syncthreads();
      compute_tile(0);
      __syncthreads();
    }
  }

  for (int i = 0; i < FM; ++i) {
    for (int j = 0; j < FN; ++j) {
      const int gm = block_m + warp_m + i * WMMA_M;
      const int gn = block_n + warp_n + j * WMMA_N;
      if (gm + WMMA_M <= M && gn + WMMA_N <= N)
        wmma::store_matrix_sync(&C[(long)gm * N + gn], acc[i][j], N, wmma::mem_row_major);
    }
  }
}

#define LAUNCH(BM, BN, BK, WM, WN, DB)                                        \
  do {                                                                        \
    dim3 grid((M + (BM) - 1) / (BM), (N + (BN) - 1) / (BN));                   \
    dim3 block((WM) * (WN) * 32);                                             \
    mm_tf32<BM, BN, BK, WM, WN, DB><<<grid, block>>>(                         \
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K); \
  } while (0)

torch::Tensor mm_cuda(torch::Tensor a, torch::Tensor b, int64_t cfg) {
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous inputs required");
  int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  switch (cfg) {
    case 0: LAUNCH(128, 128, 32,  2, 2, false); break;   //   4 warps, 64x64 per warp
    case 1: LAUNCH(128, 128, 32,  4, 2, false); break;   //   8 warps, 32x64
    case 2: LAUNCH(128, 128, 32,  2, 4, false); break;   //   8 warps, 64x32
    case 3: LAUNCH(128,  64, 32,  2, 2, false); break;
    case 4: LAUNCH( 64, 128, 32,  2, 2, false); break;
    case 5: LAUNCH( 64,  64, 32,  2, 2, false); break;
    case 6: LAUNCH(128, 128, 16,  2, 2, true ); break;   // double-buffered, 32KB shared
    case 7: LAUNCH(128, 128, 16,  4, 2, true ); break;
    case 8: LAUNCH(128,  64, 32,  2, 2, true ); break;   // 2*(128*32+32*64)*4 = 48KB
    case 9: LAUNCH( 64,  64, 32,  2, 2, true ); break;
    default: TORCH_CHECK(false, "unknown cfg");
  }
  // A launch failure must not be discovered later as a "wrong answer": surface it here.
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
  return c;
}
"""
CUDA_CONFIGS = [
    (0, "BM128 BN128 BK32 warps2x2"), (1, "BM128 BN128 BK32 warps4x2"),
    (2, "BM128 BN128 BK32 warps2x4"), (3, "BM128 BN64  BK32 warps2x2"),
    (4, "BM64  BN128 BK32 warps2x2"), (5, "BM64  BN64  BK32 warps2x2"),
    (6, "BM128 BN128 BK16 warps2x2 dbuf"), (7, "BM128 BN128 BK16 warps4x2 dbuf"),
    (8, "BM128 BN64  BK32 warps2x2 dbuf"), (9, "BM64  BN64  BK32 warps2x2 dbuf"),
]


def sweep(label, run, configs, ref, ref_absmax):
    """Time every config that is CORRECT; a wrong one is excluded, never reported."""
    best = None
    for key, name in configs:
        try:
            out = run(key)
        except Exception as exc:
            print("  %-34s SKIP (%s)" % (name, str(exc).split("\n")[0][:70]))
            continue
        err = (out - ref).abs().max().item() / ref_absmax
        if not (err <= REL_TOL):
            # Rule 5: this is excluded from the comparison, not reported as a slow/fast result.
            print("  %-34s WRONG (rel err %.2e) -- excluded" % (name, err))
            continue
        med, _ = timed(lambda: run(key), trials=30, warmup=10)
        print("  %-34s %.4f ms   (rel err %.2e)" % (name, med, err))
        if best is None or med < best[0]:
            best = (med, key, name)
    if best is None:
        print("  no %s config was both correct and runnable" % label)
    return best


def main() -> int:
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    print("device: %s sm_%d%d" % (p.name, p.major, p.minor))
    print("problem: %dx%dx%d fp32 matmul, tf32 tensor cores + fp32 accumulate on BOTH sides"
          % (M, N, K))
    print()

    a = torch.randn(M, K, device=DEV).contiguous()
    b = torch.randn(K, N, device=DEV).contiguous()

    torch.backends.cuda.matmul.allow_tf32 = True
    ref = a @ b                      # same precision path as both candidates
    ref_absmax = ref.abs().max().item()

    print("building the CUDA extension (nvcc, 10 template instantiations)...")
    ext = load_inline(
        name="step8_mm_v2",
        cpp_sources="torch::Tensor mm_cuda(torch::Tensor a, torch::Tensor b, int64_t cfg);",
        cuda_sources=CUDA_SRC, functions=["mm_cuda"], verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math",
                           "-arch=sm_%d%d" % (p.major, p.minor)])
    print("built.\n")

    print("--- Triton sweep (%d configs) ---" % len(TRITON_CONFIGS))
    t_best = sweep(
        "Triton",
        lambda cfg: triton_mm(a, b, *cfg),
        [(c, "BM%-3d BN%-3d BK%-2d nw=%d ns=%d" % c) for c in TRITON_CONFIGS],
        ref, ref_absmax)

    print("\n--- hand-written CUDA/wmma sweep (%d configs) ---" % len(CUDA_CONFIGS))
    c_best = sweep("CUDA", lambda cfg: ext.mm_cuda(a, b, cfg), CUDA_CONFIGS,
                   ref, ref_absmax)

    if t_best is None or c_best is None:
        print("\nCANNOT COMPARE: one side had no correct config.")
        return 1

    # --- full timing at each side's own best config, same function, same trial count ----
    t_med, t_mean = timed(lambda: triton_mm(a, b, *t_best[1]))
    c_med, c_mean = timed(lambda: ext.mm_cuda(a, b, c_best[1]))
    torch_med, torch_mean = timed(lambda: a @ b)

    print("\n=== our timing convention (CUDA events + per-trial L2 flush, 100 trials) ===")
    print("  torch (cuBLAS)   median %.4f ms   mean %.4f ms" % (torch_med, torch_mean))
    print("  Triton   best    median %.4f ms   mean %.4f ms   [%s]"
          % (t_med, t_mean, t_best[2]))
    print("  hand CUDA best   median %.4f ms   mean %.4f ms   [%s]"
          % (c_med, c_mean, c_best[2]))
    print()
    print("  CUDA vs Triton : %.3fx   (>1 means the hand-written CUDA is faster)"
          % (t_med / c_med))
    print("  Triton vs torch: %.3fx" % (torch_med / t_med))
    print("  CUDA   vs torch: %.3fx" % (torch_med / c_med))
    print()
    print("SCOPE: one hand-written pair, one shape, one card, and a hand-written kernel is only")
    print("as good as the hand that wrote it. This bounds the prize for backend work on this")
    print("class of kernel; it does NOT establish that either backend is faster in general.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
