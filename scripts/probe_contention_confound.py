"""Is the contention diagnostic confounded by kernel DURATION?

Contradictory results so far, from my own two runs:
  probe_throttle_v2:      torch copy (BW-bound)  +19.5%,  cuBLAS gemm 4096 (CMP)  +5.5%   <- as predicted
  probe_two_perturbations: triton elementwise (BW) +11.4%, triton gemm 2048       +82.0%  <- REVERSED

Two candidate explanations, and they have different consequences:
  (a) DURATION CONFOUND. The reversal correlates with runtime: the kernels that degraded most
      were the SHORT ones (0.0287 ms gemm256 -> +443%, 0.3021 ms gemm2048 -> +82%), while the
      long ones barely moved (2.58 ms gemm4096 -> +5.5%). A hog that occupies the memory
      system for a fixed interval hurts a short kernel proportionally more. If this is the
      cause, contention is NOT a bottleneck diagnostic -- it is a duration meter.
  (b) MISLABELED CONTROL. gemm2048 read "104% of the fp32 ceiling", which is impossible.
      tl.dot defaults to tf32, so its real ceiling is 88.1 not 54.8 TFLOP/s -> 65%, i.e. that
      control was never compute-saturated. Same error class as my earlier one.

This script separates them: ONE kernel, ONE precision, sizes swept so duration varies over
two orders of magnitude while the bottleneck stays fixed. If degradation tracks duration,
explanation (a) holds and the diagnostic is dead as written.
"""
import torch
import triton
import triton.language as tl

dev = torch.device("cuda")
torch.manual_seed(0)
DRAM = 924.1
CEIL = {"ieee": 54.8, "tf32": 88.1}


@triton.jit
def elem_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) * tl.load(y_ptr + offs, mask=m), mask=m)


@triton.jit
def mm_k(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         PREC: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def timeit(fn, n=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


hog_a = torch.randn(1 << 24, device=dev)
hog_b = torch.empty_like(hog_a)
hog_stream = torch.cuda.Stream()


def contended(fn, rounds=25, hog_iters=8):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        with torch.cuda.stream(hog_stream):
            for _ in range(hog_iters):
                hog_b.copy_(hog_a)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


print("=" * 92)
print("A. BANDWIDTH-BOUND family: same kernel, duration swept 60x. Bottleneck FIXED.")
print("=" * 92)
print("%-12s %10s %9s %10s %10s" % ("elements", "median ms", "%of DRAM", "contended", "degrade"))
for shift in (20, 22, 24, 26):
    n = 1 << shift
    x, y, o = (torch.randn(n, device=dev) for _ in range(3))
    g = triton.cdiv(n, 1024)

    def run(g=g, n=n, x=x, y=y, o=o):
        elem_k[(g,)](x, y, o, n, BLOCK=1024)

    base = timeit(run)
    util = 100 * (3 * n * 4) / (base / 1e3) / 1e9 / DRAM
    cont = contended(run)
    print("%-12s %10.4f %8.0f%% %10.4f %+9.1f%%"
          % (f"{n:,}", base, util, cont, 100 * (cont - base) / base))
    del x, y, o

print()
print("=" * 92)
print("B. COMPUTE-BOUND family: strict-ieee dot so the fp32 ceiling really applies.")
print("=" * 92)
print("%-12s %10s %9s %10s %10s" % ("M", "median ms", "%of ieee", "contended", "degrade"))
for M in (512, 1024, 2048, 4096):
    a, b = torch.randn(M, M, device=dev), torch.randn(M, M, device=dev)
    c = torch.empty(M, M, device=dev)
    grid = (triton.cdiv(M, 64), triton.cdiv(M, 64))

    def run(grid=grid, a=a, b=b, c=c, M=M):
        mm_k[grid](a, b, c, M, M, M, BM=64, BN=64, BK=32, PREC="ieee")

    base = timeit(run, n=25, warmup=8)
    util = 100 * (2 * M ** 3) / (base / 1e3) / 1e12 / CEIL["ieee"]
    cont = contended(run, rounds=20)
    print("%-12s %10.4f %8.0f%% %10.4f %+9.1f%%"
          % (M, base, util, cont, 100 * (cont - base) / base))
    del a, b, c

print()
print("=" * 92)
print("VERDICT")
print("=" * 92)
print("Read the two tables COLUMNWISE, not rowwise:")
print("  If degradation tracks DURATION within each family (short = big degradation),")
print("    the diagnostic measures duration and is UNUSABLE as a bottleneck signal.")
print("  If degradation is roughly CONSTANT within a family and DIFFERS between families,")
print("    it is a real bottleneck signal and the earlier reversal was the mislabeled control.")
print("  Any comparison between kernels of different duration must be duration-matched.")
