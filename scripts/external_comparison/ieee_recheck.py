"""The ieee attention number disagreed across two runs of the SAME tile: 6.737 ms in the
projection/attention split, 12.339 ms in the tile sweep. Both are 100/30-sample medians on an idle
card, so one of the two measurements is not measuring what its label says. Resolve it before any of
it is reported.

The suspicion is the INPUT, not the kernel. The split fed `qkv3 = torch.randn(...)`; a causal
softmax over randn data is numerically tame. The sweep fed randn too -- but the two runs differ in
whether the ieee kernel was compiled with stages=1 or stages=2 and in what ran before it in the
same process. So: re-measure the ieee tile alone, repeatedly, in a fresh process, interleaved with
the tf32 tile as an in-run control, and print every sample's spread rather than only the median.
"""
import math, importlib.util
import torch, triton

P = "/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py"
spec = importlib.util.spec_from_file_location("cand", P)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def stats(fn, n=100, warmup=20):
    with torch.no_grad():
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s = []
        for _ in range(n):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(b))
    s.sort()
    return s[len(s)//2], s[0], s[-1]

dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
hs_pad = max(16, triton.next_power_of_2(hs))

def run(mode, bm, bn, nw, ns, dt, label):
    qkv3 = torch.randn(B, T, 3 * C, device=dev, dtype=dt)
    yout = torch.empty((B, T, C), device=dev, dtype=dt)
    f = lambda: mod._attn_fwd[(B * NH, triton.cdiv(T, bm))](
        qkv3, yout, T, 1.0 / math.sqrt(hs),
        qkv3.stride(0), qkv3.stride(1), yout.stride(0), yout.stride(1),
        NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C,
        BLOCK_M=bm, BLOCK_N=bn, MODE=mode, num_warps=nw, num_stages=ns)
    m, lo, hi = stats(f)
    print(f"  {label:34s} median {m:7.3f}  min {lo:7.3f}  max {hi:7.3f}")
    return m

print("### three interleaved passes: ieee(64,32,8,1), ieee(64,32,8,2), tf32(128,32,8,2) ###")
for p in range(3):
    print(f"pass {p}")
    run(0, 64, 32, 8, 1, torch.float32, "ieee BM64 BN32 w8 stages=1")
    run(0, 64, 32, 8, 2, torch.float32, "ieee BM64 BN32 w8 stages=2")
    run(1, 128, 32, 8, 2, torch.float32, "tf32 BM128 BN32 w8 stages=2 (ctrl)")
