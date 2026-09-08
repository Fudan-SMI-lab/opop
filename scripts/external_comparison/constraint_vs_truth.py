"""agent 写的 dtype-aware shared-memory 约束,算出来的字节数是否等于 Triton 真实需要的?

这是决定改进方案的分水岭:
  - 如果约束基本准确,那 L3:43 的失败是别的原因(约束缺失/未覆盖某个 kernel),方案 = 补覆盖检查;
  - 如果约束系统性偏小,那 agent 的手算不可信,方案 = 用编译期真值(P1 已有 compile_probe)
    替代手写约束,而不是继续教 agent 怎么算。

判据用 P1 的 compile_probe:Triton 编译后 metadata.shared 就是运行期 Required 的字节数(已 6/6 验证)。
拿 L3:43 sweep 里那些已知真值的配置来比。
"""
import re, pathlib, sys, itertools
import torch, triton

sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
SRC = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py")
BASE = SRC.read_text()

def build(compute, io, attn):
    am, an, aw, as_ = attn
    vals = {"ATTN_BLOCK_M": am, "ATTN_BLOCK_N": an, "ATTN_NUM_WARPS": aw,
            "ATTN_NUM_STAGES": as_, "COMPUTE_DTYPE": f"'{compute}'", "IO_DTYPE": f"'{io}'"}
    s = BASE
    for k, v in vals.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s)
        assert n == 1, k
    return s

# The shape of constraint the agents actually wrote for this task's attention kernel
# (cand-969997e3): NUM_STAGES * 2 * BLOCK_N * 64 * w, w=2 for fp16/bf16, 4 for tf32/ieee.
def agent_formula(bn, stages, compute):
    w = 2 if compute in ("fp16", "bf16") else 4
    return stages * 2 * bn * 64 * w

# A second shape, from cand-6e7f7b58, which also counts the Q tile:
def agent_formula2(bm, bn, stages, compute):
    w = 2 if compute in ("fp16", "bf16") else 4
    return (2 * bm * 64 + 2 * stages * 2 * bn * 64) * (w // 2)

import importlib.util
dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
hs_pad = max(16, triton.next_power_of_2(hs))

print(f"{'compute':7s} {'BM':>4s} {'BN':>4s} {'stg':>3s} {'TRUE':>8s} "
      f"{'formula1':>9s} {'ratio':>6s} {'formula2':>9s} {'ratio':>6s}")
rows = []
for compute in ("fp16", "tf32"):
    io = "fp16" if compute == "fp16" else "fp32"
    for bm, bn, stages in itertools.product((64, 128), (32, 64), (1, 2, 3)):
        p = pathlib.Path(f"/root/autodl-tmp/ext-eval/fp32/cvt_{compute}_{bm}_{bn}_{stages}.py")
        p.write_text(build(compute, io, (bm, bn, 8, stages)))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  load failed {compute} {bm} {bn} {stages}: {type(e).__name__}")
            continue
        # compile WITHOUT launching, the P1 way: warmup=True returns the compiled kernel
        try:
            k = mod._attn_fwd.warmup(
                torch.empty(1, device=dev), torch.empty(1, device=dev), T, 1.0,
                1, 1, 1, 1,
                NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C,
                BLOCK_M=bm, BLOCK_N=bn,
                MODE={"fp16": 2, "tf32": 1}[compute],
                num_warps=8, num_stages=stages, grid=(1, 1))
            true_shared = k.metadata.shared
        except Exception as e:
            print(f"  probe failed {compute} {bm} {bn} {stages}: "
                  f"{type(e).__name__}: {str(e)[:70]}")
            continue
        f1 = agent_formula(bn, stages, compute)
        f2 = agent_formula2(bm, bn, stages, compute)
        print(f"{compute:7s} {bm:4d} {bn:4d} {stages:3d} {true_shared:8d} "
              f"{f1:9d} {f1/true_shared:6.2f} {f2:9d} {f2/true_shared:6.2f}")
        rows.append((true_shared, f1, f2))

if rows:
    import statistics
    print(f"\n  n={len(rows)}")
    for name, idx in (("formula1", 1), ("formula2", 2)):
        rs = [r[idx] / r[0] for r in rows]
        under = sum(1 for r in rs if r < 0.95)
        print(f"  {name}: ratio median {statistics.median(rs):.2f} "
              f"min {min(rs):.2f} max {max(rs):.2f}  "
              f"UNDER-estimates (would admit a config that dies) in {under}/{len(rs)}")
