"""Verify the derived numbers in the write-up that were computed rather than measured.

Claims under test:
  (a) "ours 9.58 ms projections at ieee / 3.96 at tf32"  -- I asserted these from the GEMM sweep's
      per-shape bests; they must be the SUM of the two shapes' best times, re-measured together.
  (b) "our Triton GEMM at ieee reaches 59% of the fp32 roof (32.6 TFLOP/s)"
  (c) "attention is ~51.5 GFLOP after causal halving" -> their 1.3-1.7 ms = 30-41 TFLOP/s
  (d) bf16 total 4.124 ms -- carried from an earlier session, never re-measured on this box.
"""
import importlib.util, pathlib, re
import torch

# (c) FLOP arithmetic, no GPU needed
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
qk = 2 * B * NH * T * T * hs        # Q@K^T
av = 2 * B * NH * T * T * hs        # P@V
attn_flop_full = qk + av
attn_flop_causal = attn_flop_full / 2
proj_flop = 2 * (B*T) * C * (3*C) + 2 * (B*T) * C * C
print(f"(c) attention FLOP: full {attn_flop_full/1e9:.1f} GFLOP, "
      f"causal-halved {attn_flop_causal/1e9:.1f} GFLOP")
print(f"    projections    : {proj_flop/1e9:.1f} GFLOP   "
      f"total {(attn_flop_causal+proj_flop)/1e9:.1f} GFLOP  "
      f"proj share {proj_flop/(attn_flop_causal+proj_flop)*100:.1f}%")
for ms in (1.264, 1.705):
    print(f"    their attention at {ms:.3f} ms -> {attn_flop_causal/(ms*1e-3)/1e12:.1f} TFLOP/s "
          f"= {attn_flop_causal/(ms*1e-3)/1e12/54.95*100:.0f}% of fp32 roof 54.95")
print(f"    ours 5.225 ms -> {attn_flop_causal/(5.225e-3)/1e12:.1f} TFLOP/s "
      f"= {attn_flop_causal/(5.225e-3)/1e12/54.95*100:.0f}% of roof")

def med(fn, n=100, warmup=20):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s=[]
        for _ in range(n):
            a,b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); s.append(a.elapsed_time(b))
    s.sort(); return s[len(s)//2]

# (a)+(b): re-measure our Triton GEMM at its per-precision best tile, both shapes, summed.
SRC = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py")
spec = importlib.util.spec_from_file_location("cand", SRC)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import triton
dev = torch.device("cuda")
M, K = B*T, C
BEST_LIN = {"ieee": (128,256,32,8,2), "tf32": (64,128,32,4,3), "fp16": (128,128,32,8,2),
            "bf16": (128,128,32,8,2)}
print()
for mname, (bm,bn,bk,nw,ns) in BEST_LIN.items():
    torch.backends.cuda.matmul.allow_tf32 = (mname == "tf32")
    mode = {"ieee":0,"tf32":1,"fp16":2,"bf16":3}[mname]
    dt = {"fp16":torch.float16,"bf16":torch.bfloat16}.get(mname, torch.float32)
    tot_ours = tot_cub = 0.0
    for N in (3*C, C):
        a = torch.randn(M, K, device=dev, dtype=dt)
        w = torch.randn(N, K, device=dev, dtype=dt)
        bb = torch.randn(N, device=dev, dtype=torch.float32)
        y = torch.empty(M, N, device=dev, dtype=torch.float32)
        t = med(lambda: mod._linear_fwd[(triton.cdiv(N,bn), triton.cdiv(M,bm))](
            a, w, bb, y, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, MODE=mode,
            num_warps=nw, num_stages=ns), n=40, warmup=10)
        c = med(lambda: torch.nn.functional.linear(a, w, bb.to(dt)), n=40, warmup=10)
        tot_ours += t; tot_cub += c
    roof = {"ieee":54.95,"tf32":89.06,"fp16":164.4,"bf16":166.7}[mname]
    tf = proj_flop/(tot_ours*1e-3)/1e12
    print(f"(a) {mname:5s} our GEMM both shapes {tot_ours:6.3f} ms | cuBLAS {tot_cub:6.3f} ms "
          f"| ratio {tot_ours/tot_cub:.2f}x | ours {tf:5.1f} TFLOP/s = {tf/roof*100:.0f}% of {roof}")

# (d) bf16 end-to-end total, re-measured
BASE = SRC.read_text()
def build(compute, io, lin, attn):
    bm,bn,bk,lw,ls = lin; am,an,aw,as_ = attn
    vals = {"ATTN_BLOCK_M":am,"ATTN_BLOCK_N":an,"ATTN_NUM_WARPS":aw,"ATTN_NUM_STAGES":as_,
            "COMPUTE_DTYPE":f"'{compute}'","IO_DTYPE":f"'{io}'","LINEAR_BLOCK_M":bm,
            "LINEAR_BLOCK_N":bn,"LINEAR_BLOCK_K":bk,"LINEAR_NUM_WARPS":lw,"LINEAR_NUM_STAGES":ls}
    s = BASE
    for k,v in vals.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s); assert n == 1, k
    return s
print()
x = torch.randn(B, T, C, device=dev)
for compute, io, lin, attn in (("bf16","bf16",(128,128,32,8,2),(128,32,4,2)),
                               ("fp16","fp16",(128,128,32,8,2),(128,64,8,2))):
    p = pathlib.Path(f"/root/autodl-tmp/ext-eval/fp32/vc_{compute}.py")
    p.write_text(build(compute, io, lin, attn))
    sp = importlib.util.spec_from_file_location(p.stem, p)
    m2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(m2)
    mm = m2.ModelNew(C, NH, 0.0, 0.0, T).to(dev).eval()
    print(f"(d) {compute} end-to-end total: {med(lambda: mm(x)):.3f} ms")
