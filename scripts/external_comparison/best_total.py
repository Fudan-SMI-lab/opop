"""End-to-end total our Triton candidate reaches when the linear and attention tiles are chosen
PER PRECISION, versus what it actually shipped (one tile for every precision).

This is the number that decides "Triton limitation vs our tuning was insufficient". The pieces are
already measured; this assembles them into a real end-to-end run so the answer is not a sum of
isolated kernels (which would ignore launch gaps and the dtype casts in `_linear`).

Per-precision optima from the two sweeps on this box:
  linear    ieee (128,256,32,w8,s2)   tf32 (64,128,32,w4,s3)   fp16 (128,128,32,w8,s2)
  attention ieee (64,32,w8,s2)        tf32 (128,32,w8,s2)      fp16 (128,64,w8,s2)
Shipped for every precision: linear (128,128,32,w8,s2), attention (128,64,w8,s2).
"""
import importlib.util, pathlib, re, sys
import torch

SRC = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py")
BASE = SRC.read_text()

# (compute, io, linear bm/bn/bk/warps/stages, attn bm/bn/warps/stages)
CONFIGS = [
    ("ieee shipped", "ieee", "fp32", (128, 128, 32, 8, 2), (128, 64, 8, 2)),
    ("ieee retuned", "ieee", "fp32", (128, 256, 32, 8, 2), (64, 32, 8, 2)),
    ("tf32 shipped", "tf32", "fp32", (128, 128, 32, 8, 2), (128, 64, 8, 2)),
    ("tf32 retuned", "tf32", "fp32", (64, 128, 32, 4, 3), (128, 32, 8, 2)),
    ("fp16 shipped", "fp16", "fp16", (128, 128, 32, 8, 2), (128, 64, 8, 2)),
    ("fp16 retuned", "fp16", "fp16", (128, 128, 32, 8, 2), (128, 64, 8, 2)),
]

def build(compute, io, lin, attn):
    bm, bn, bk, lw, ls = lin
    am, an, aw, as_ = attn
    vals = {"ATTN_BLOCK_M": am, "ATTN_BLOCK_N": an, "ATTN_NUM_WARPS": aw, "ATTN_NUM_STAGES": as_,
            "COMPUTE_DTYPE": f"'{compute}'", "IO_DTYPE": f"'{io}'",
            "LINEAR_BLOCK_M": bm, "LINEAR_BLOCK_N": bn, "LINEAR_BLOCK_K": bk,
            "LINEAR_NUM_WARPS": lw, "LINEAR_NUM_STAGES": ls}
    src = BASE
    for k, v in vals.items():
        src, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", src)
        assert n == 1, f"{k} replaced {n} times"
    return src

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
out = {}
for label, compute, io, lin, attn in CONFIGS:
    path = pathlib.Path(f"/root/autodl-tmp/ext-eval/fp32/tot_{label.replace(' ', '_')}.py")
    path.write_text(build(compute, io, lin, attn))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    torch.backends.cuda.matmul.allow_tf32 = (compute == "tf32")
    try:
        spec.loader.exec_module(mod)
        m = mod.ModelNew(C, NH, 0.0, 0.0, T).to(dev).eval()
        t = med(lambda: m(x))
        out[label] = t
        print(f"  {label:14s} linear={lin} attn={attn}  ->  {t:7.3f} ms")
    except Exception as e:
        print(f"  {label:14s} FAILED {type(e).__name__}: {str(e)[:110]}")

print("\n### summary ###")
for reg in ("ieee", "tf32", "fp16"):
    s, r = out.get(f"{reg} shipped"), out.get(f"{reg} retuned")
    if s and r:
        print(f"  {reg}: shipped {s:7.3f} -> retuned {r:7.3f} ms   ({(s/r-1)*100:+.1f}% recovered)")
print("\n  external CUDA on this box: ieee 8.893 ms, tf32 5.529 ms (its only two regimes)")
