"""At equal precision CUDA wins. Is that a Triton limitation, or our tuning?

Split the time. Both designs use cuBLAS at::linear / a Triton GEMM for the projections and a
fused flash kernel for attention. If the gap is in the PROJECTIONS, it is a library-vs-Triton
GEMM question. If it is in the ATTENTION, it is about the flash kernel.
"""
import importlib.util, os, sys
import torch
os.environ.setdefault("EXT_KERNEL_CU", "/root/autodl-tmp/ext-eval/L3_43/best_kernel.cu")
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/KernelBench/src")
ref_ctx = {}
exec(compile(open("/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/"
                  "43_MinGPTCausalAttention.py").read(), "ref", "exec"), ref_ctx)
dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8

def med(fn, n=60, warmup=15):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(n):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(b))
    s.sort(); return s[len(s)//2]

x = torch.randn(B, T, C, device=dev)
wq = torch.randn(3*C, C, device=dev); bq = torch.randn(3*C, device=dev)
wp = torch.randn(C, C, device=dev); bp = torch.randn(C, device=dev)
prev = torch.backends.cuda.matmul.allow_tf32
for label, flag in (("ieee", False), ("tf32", True)):
    torch.backends.cuda.matmul.allow_tf32 = flag
    qkv = med(lambda: torch.nn.functional.linear(x, wq, bq))
    y = torch.randn(B, T, C, device=dev)
    proj = med(lambda: torch.nn.functional.linear(y, wp, bp))
    print(f"  {label:5s}  cuBLAS QKV {qkv:6.3f} ms + proj {proj:6.3f} ms = "
          f"{qkv+proj:6.3f} ms of projections alone")
torch.backends.cuda.matmul.allow_tf32 = prev
print()
print("  Compare with the measured totals:")
print("    external CUDA ieee 8.893   tf32 5.529")
print("    our Triton    ieee 15.975  tf32 6.129")
