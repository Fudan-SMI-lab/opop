"""agent 真实声明的 shared-memory 约束 vs 编译期真值:一个 2x2 混淆矩阵。

前一个实验证明约束低估真值,但那只说明"约束放行了会死的配置"。改进方案还取决于另一个方向:
约束是否也**误禁**了本来能跑的配置 —— 那种损失是静默的(没有失败,只有一个被悄悄缩小的搜索空间)。

所以这里用 guard 自己的求值器跑 agent 真实写下的约束表达式(从 events.jsonl 读出,不是我编的),
在同一个 tile 网格上与"编译期能否装下"对照:

  admit&die  = 约束放行、真值超限   -> 浪费 trial(P1 的 screen 现在能拦,但 trial 已花掉)
  forbid&fit = 约束拒绝、真值能装   -> 静默丢失的搜索空间(没有任何机制能发现)
"""
import itertools, pathlib, re, sys
import torch, triton

sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.paramspace.guard import eval_constraint
from kernel_optimizer.models.core import DeviceLimits

DEV = DeviceLimits(max_regs_per_thread=255, max_shared_bytes_static=49152,
                   max_shared_bytes_optin=101376, max_threads_per_block=1024)
LIMIT = 101376

# L3:43 cand-969997e3's ACTUAL declared constraint, verbatim from events.jsonl.
C43 = ('(COMPUTE_DTYPE == "fp16" or COMPUTE_DTYPE == "bf16") and '
       'NUM_STAGES * 2 * BLOCK_N * 64 * 2 <= MAX_SHARED_BYTES_OPTIN '
       'or (COMPUTE_DTYPE == "tf32" or COMPUTE_DTYPE == "ieee") and '
       'NUM_STAGES * 2 * BLOCK_N * 64 * 4 <= MAX_SHARED_BYTES_OPTIN')

SRC = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py")
BASE = SRC.read_text()
import importlib.util
dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
hs_pad = max(16, triton.next_power_of_2(hs))

def true_shared(compute, io, bm, bn, stages):
    p = pathlib.Path(f"/root/autodl-tmp/ext-eval/fp32/cf_{compute}_{bm}_{bn}_{stages}.py")
    s = BASE
    for k, v in {"ATTN_BLOCK_M": bm, "ATTN_BLOCK_N": bn, "ATTN_NUM_WARPS": 8,
                 "ATTN_NUM_STAGES": stages, "COMPUTE_DTYPE": f"'{compute}'",
                 "IO_DTYPE": f"'{io}'"}.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s); assert n == 1, k
    p.write_text(s)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    k = mod._attn_fwd.warmup(
        torch.empty(1, device=dev), torch.empty(1, device=dev), T, 1.0, 1, 1, 1, 1,
        NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C, BLOCK_M=bm, BLOCK_N=bn,
        MODE={"fp16": 2, "bf16": 3, "tf32": 1, "ieee": 0}[compute],
        num_warps=8, num_stages=stages, grid=(1, 1))
    return k.metadata.shared

cells = {"admit&fit": 0, "admit&die": 0, "forbid&fit": 0, "forbid&die": 0}
rows = []
for compute in ("fp16", "tf32", "ieee"):
    io = "fp16" if compute == "fp16" else "fp32"
    for bm, bn, stages in itertools.product((64, 128), (32, 64), (1, 2, 3)):
        try:
            t = true_shared(compute, io, bm, bn, stages)
        except Exception as e:
            continue
        env = {"COMPUTE_DTYPE": compute, "NUM_STAGES": stages,
               "BLOCK_N": bn, "BLOCK_M": bm}
        try:
            admitted = bool(eval_constraint(C43, {**DEV.as_env(), **env}))
        except Exception as e:
            print("constraint eval failed:", e); sys.exit(1)
        fits = t <= LIMIT
        key = ("admit" if admitted else "forbid") + "&" + ("fit" if fits else "die")
        cells[key] += 1
        rows.append((compute, bm, bn, stages, t, admitted, fits, key))

print(f"{'dtype':6s} {'BM':>4s} {'BN':>4s} {'stg':>3s} {'TRUE':>8s} "
      f"{'constraint':>10s} {'reality':>8s}  verdict")
for compute, bm, bn, stages, t, adm, fits, key in rows:
    flag = "  <-- WASTED TRIAL" if key == "admit&die" else (
           "  <-- LOST CONFIG" if key == "forbid&fit" else "")
    print(f"{compute:6s} {bm:4d} {bn:4d} {stages:3d} {t:8d} "
          f"{'admit' if adm else 'forbid':>10s} {'fits' if fits else 'dies':>8s}{flag}")
n = len(rows)
print(f"\n  n={n}")
for k, v in cells.items():
    print(f"    {k:11s} {v:3d}  ({v/n*100:.0f}%)")
print(f"\n  the constraint's agreement with reality: "
      f"{(cells['admit&fit'] + cells['forbid&die'])/n*100:.0f}%")
