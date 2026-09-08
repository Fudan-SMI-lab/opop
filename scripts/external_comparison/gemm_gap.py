"""Is our Triton GEMM losing to cuBLAS because Triton cannot, or because ours is under-written?

Our candidate's `_linear_fwd` uses a plain 2D grid: pid_n = program_id(0), pid_m = program_id(1).
The textbook Triton matmul instead SWIZZLES the program id into groups of GROUP_M rows so that
concurrently-resident CTAs share B tiles in L2. That single change is the standard 10-25% on
Triton GEMMs, and our parameter space never had a GROUP_M knob -- the tuner could not have found
it. If the swizzle closes the gap to cuBLAS, the gap is our search space, not Triton.

Same shapes as the task: qkv is (65536, 768) x (768, 2304); proj is (65536, 768) x (768, 768).
"""
import torch, triton, triton.language as tl

@triton.jit
def _plain(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
           MODE: tl.constexpr):
    """Byte-for-byte the shape of our candidate's kernel: no swizzle."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_valid, n_valid = offs_m < M, offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_valid = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_valid[:, None] & k_valid[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=n_valid[None, :] & k_valid[:, None], other=0.0)
        if MODE == 1:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        elif MODE == 0:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=m_valid[:, None] & n_valid[None, :])

@triton.jit
def _swizzled(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, MODE: tl.constexpr):
    """Identical arithmetic; only the CTA->tile mapping differs (grouped-M for L2 reuse)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_valid, n_valid = offs_m < M, offs_n < N
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_valid = offs_k < (K - k0)
        a = tl.load(x_ptrs, mask=m_valid[:, None] & k_valid[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=n_valid[None, :] & k_valid[:, None], other=0.0)
        if MODE == 1:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        elif MODE == 0:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=m_valid[:, None] & n_valid[None, :])

def med(fn, n=100, warmup=20):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(n):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(b))
    s.sort(); return s[len(s)//2]

dev = torch.device("cuda")
M, K = 65536, 768
MODES = {"ieee": (0, torch.float32), "tf32": (1, torch.float32),
         "fp16": (2, torch.float16), "bf16": (3, torch.bfloat16)}
# The candidate's own tile for the linear, plus a couple of neighbours the tuner did explore.
TILES = [(128, 128, 32, 8, 2), (128, 64, 32, 4, 3), (64, 128, 32, 4, 3), (128, 256, 32, 8, 2)]
GROUPS = [1, 4, 8]

for N in (2304, 768):
    print(f"\n########  GEMM ({M} x {K}) x ({K} x {N})  "
          f"{2*M*N*K/1e9:.1f} GFLOP  ########")
    for mname, (mode, dt) in MODES.items():
        torch.backends.cuda.matmul.allow_tf32 = (mname == "tf32")
        a = torch.randn(M, K, device=dev, dtype=dt)
        w = torch.randn(N, K, device=dev, dtype=dt)
        b = torch.randn(N, device=dev, dtype=torch.float32)
        y = torch.empty(M, N, device=dev, dtype=torch.float32)
        cub = med(lambda: torch.nn.functional.linear(a, w, b.to(dt)))
        best_plain = best_sw = None
        for (bm, bn, bk, nw, ns) in TILES:
            try:
                t = med(lambda: _plain[(triton.cdiv(N, bn), triton.cdiv(M, bm))](
                    a, w, b, y, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, MODE=mode,
                    num_warps=nw, num_stages=ns), n=40, warmup=10)
                if best_plain is None or t < best_plain[0]: best_plain = (t, (bm, bn, bk, nw, ns))
            except Exception: pass
            for g in GROUPS:
                try:
                    t = med(lambda: _swizzled[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                        a, w, b, y, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                        GROUP_M=g, MODE=mode, num_warps=nw, num_stages=ns), n=40, warmup=10)
                    if best_sw is None or t < best_sw[0]: best_sw = (t, (bm, bn, bk, nw, ns), g)
                except Exception: pass
        pl = f"{best_plain[0]:.3f} {best_plain[1]}" if best_plain else "all failed"
        sw = f"{best_sw[0]:.3f} {best_sw[1]} G={best_sw[2]}" if best_sw else "all failed"
        gain = f"{best_plain[0]/best_sw[0]:.3f}x" if best_plain and best_sw else "-"
        print(f"  {mname:5s} cuBLAS {cub:7.3f} | plain {pl:34s} | swizzled {sw:38s}"
              f" | swizzle {gain} | best/cuBLAS "
              f"{min(best_plain[0], best_sw[0])/cub:.2f}x")
