"""Decompose the CUDA-vs-Triton gap at equal precision into projections + attention.

Both designs are: two projections (86% of the task's FLOPs) + a fused causal-attention kernel.
  external CUDA : cuBLAS at::linear   + hand-written flash kernel
  our Triton    : own Triton GEMM     + Triton flash kernel

So three numbers separate them: cuBLAS vs our GEMM, their flash vs our flash, and the total.
Timing our GEMM and our attention kernel separately gives the split directly; without it,
"CUDA is 1.8x faster" names no cause and cannot be acted on.

Per-regime best tiles are taken from fp32_search.log on this box (ieee 64/32/1, tf32 128/32/2):
the ieee optimum is NOT the tf32 one, and using the tf32 tile at ieee overflows shared memory.
"""
import math, sys, importlib.util
import torch, triton

BEST = {"ieee": "k_ieee_64_32_1.py", "tf32": "k_tf32_128_32_2.py"}
ROOT = "/root/autodl-tmp/ext-eval/fp32/"
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/KernelBench/src")

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

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
B, T, C, NH = 128, 512, 768, 8
torch.manual_seed(0)
x = torch.randn(B, T, C, device=dev)
M, hs = B * T, C // NH
xr = x.reshape(M, C).contiguous()
y2 = torch.randn(M, C, device=dev)

for regime, fname in BEST.items():
    mod = load_mod(ROOT + fname, f"cand_{regime}")
    torch.backends.cuda.matmul.allow_tf32 = (regime == "tf32")
    m = mod.ModelNew(C, NH, 0.0, 0.0, T).to(dev).eval()
    P = mod.PARAMS
    total = med(lambda: m(x))

    ours_qkv = med(lambda: m._linear(xr, m.c_attn.weight, m.c_attn.bias, M, P, torch.float32))
    ours_proj = med(lambda: m._linear(y2, m.c_proj.weight, m.c_proj.bias, M, P, torch.float32))
    cub_qkv = med(lambda: torch.nn.functional.linear(xr, m.c_attn.weight, m.c_attn.bias))
    cub_proj = med(lambda: torch.nn.functional.linear(y2, m.c_proj.weight, m.c_proj.bias))
    ours_gemm, cub = ours_qkv + ours_proj, cub_qkv + cub_proj

    qkv3 = torch.randn(B, T, 3 * C, device=dev)
    yout = torch.empty((B, T, C), device=dev)
    hs_pad = max(16, triton.next_power_of_2(hs))
    amode = mod._DT_MODE[P["COMPUTE_DTYPE"]]
    def attn():
        mod._attn_fwd[(B * NH, triton.cdiv(T, P["ATTN_BLOCK_M"]))](
            qkv3, yout, T, 1.0 / math.sqrt(hs),
            qkv3.stride(0), qkv3.stride(1), yout.stride(0), yout.stride(1),
            NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C,
            BLOCK_M=P["ATTN_BLOCK_M"], BLOCK_N=P["ATTN_BLOCK_N"], MODE=amode,
            num_warps=P["ATTN_NUM_WARPS"], num_stages=P["ATTN_NUM_STAGES"])
    attn_ms = med(attn)

    print(f"=== {regime}  ({fname}, BM={P['ATTN_BLOCK_M']} BN={P['ATTN_BLOCK_N']} "
          f"stages={P['ATTN_NUM_STAGES']}) ===")
    print(f"  our total                 {total:8.3f} ms")
    print(f"  our Triton GEMM  qkv+proj {ours_gemm:8.3f} ms  ({ours_qkv:.3f} + {ours_proj:.3f})")
    print(f"  cuBLAS           qkv+proj {cub:8.3f} ms  ({cub_qkv:.3f} + {cub_proj:.3f})"
          f"   -> ours is {ours_gemm/cub:.2f}x cuBLAS")
    print(f"  our Triton attention      {attn_ms:8.3f} ms  ({attn_ms/total*100:.1f}% of total)")
    print(f"  hybrid cuBLAS + our attn  {cub + attn_ms:8.3f} ms  <- what our attn would score")
    print(f"  GEMM excess over cuBLAS   {ours_gemm - cub:8.3f} ms  <- the recoverable part")
    print()
