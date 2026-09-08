"""F5 的 GPU 判据。独立运行,不覆盖 box 上任何被修改的文件。

用我本机改过的 run_compile_probe 逻辑(以 --probe-src 传入的源码文件为准)复现两件事:
  1. 批量探测 36 个 L3:43 配置,拒绝数应与 confusion.py 的真值列一致(17 拒 / 19 放行);
  2. 反向对照:把设备上限设成 1e9,应该 0 拒绝。
没有第 2 步,一个"永远拒绝"和一个"永远放行"的实现都会看起来通过。
"""
import importlib.util, itertools, pathlib, re, sys, time
import torch, triton

LIMIT = 101376
BASE = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py").read_text()
dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
hs_pad = max(16, triton.next_power_of_2(hs))
MODE = {"ieee": 0, "tf32": 1, "fp16": 2, "bf16": 3}
W = pathlib.Path("/root/autodl-tmp/ext-eval/f5probe"); W.mkdir(exist_ok=True)

def build(compute, bm, bn, stages):
    s = BASE
    for k, v in {"ATTN_BLOCK_M": bm, "ATTN_BLOCK_N": bn, "ATTN_NUM_STAGES": stages,
                 "COMPUTE_DTYPE": f"'{compute}'",
                 "IO_DTYPE": "'fp16'" if compute in ("fp16", "bf16") else "'fp32'"}.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s); assert n == 1, k
    return s

# The grid confusion.py used, so the two are directly comparable.
grid = [(c, bm, bn, st) for c in ("fp16", "tf32", "ieee")
        for bm, bn, st in itertools.product((64, 128), (32, 64), (1, 2, 3))]

t0 = time.perf_counter()
shared = {}
for i, (compute, bm, bn, st) in enumerate(grid):
    p = W / f"v{i}.py"
    p.write_text(build(compute, bm, bn, st))
    spec = importlib.util.spec_from_file_location(f"kopt_compile_probe_{i}", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        k = mod._attn_fwd.warmup(
            torch.empty(1, device=dev), torch.empty(1, device=dev), T, 1.0, 1, 1, 1, 1,
            NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C, BLOCK_M=bm, BLOCK_N=bn,
            MODE=MODE[compute], num_warps=8, num_stages=st, grid=(1, 1))
        shared[(compute, bm, bn, st)] = k.metadata.shared
    except Exception as e:
        shared[(compute, bm, bn, st)] = None
        print(f"  probe failed {compute} {bm} {bn} {st}: {type(e).__name__}")
elapsed = time.perf_counter() - t0

answered = {k: v for k, v in shared.items() if v is not None}
print(f"batched {len(grid)} configs in ONE process: {elapsed:.2f}s "
      f"({elapsed/len(grid)*1000:.0f} ms each), answered {len(answered)}/{len(grid)}")

def verdicts(limit):
    refused = [k for k, v in answered.items() if v > limit]
    return refused

refused = verdicts(LIMIT)
print(f"\nat the real limit {LIMIT}: refused {len(refused)}, admitted "
      f"{len(answered) - len(refused)}")
print("  expected from confusion.py: 17 refused / 19 admitted")
ok1 = (len(refused) == 17 and len(answered) - len(refused) == 19)
print(f"  MATCH: {ok1}")

# REVERSE CONTROL: an implementation that always refuses, or always admits, would both look
# fine on the forward check alone.
refused_huge = verdicts(10**9)
print(f"\nreverse control, limit 1e9: refused {len(refused_huge)} (expect 0)")
ok2 = len(refused_huge) == 0
print(f"  MATCH: {ok2}")

print(f"\nboth checks pass: {ok1 and ok2}")
for k in sorted(refused, key=lambda x: (x[0], x[1], x[2], x[3])):
    print(f"    refused {k[0]:5s} BM={k[1]:3d} BN={k[2]:2d} stg={k[3]} -> {answered[k]}")
