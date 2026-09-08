"""16.7s per probe kills the naive "move the screen into guard_ok". So test the design that
actually fits: ONE worker process probes MANY configurations.

The 16.7s is almost all fixed cost -- process start, torch import, CUDA context, KernelBench
import. The Triton compile of one kernel is a fraction of it. If one process can screen the whole
grid, the per-configuration cost collapses and the screen becomes affordable at ask() time
(or, better, as a one-shot pre-pass over the sampler's own grid).

Measured here: N configurations in one process, reporting both total and per-config marginal.
"""
import importlib.util, pathlib, re, time, sys, itertools
import torch, triton

BASE = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py").read_text()
W = pathlib.Path("/root/autodl-tmp/ext-eval/probebatch"); W.mkdir(exist_ok=True)
dev = torch.device("cuda")
B, T, C, NH = 128, 512, 768, 8
hs = C // NH
hs_pad = max(16, triton.next_power_of_2(hs))
MODE = {"ieee": 0, "tf32": 1, "fp16": 2, "bf16": 3}

t_start = time.perf_counter()
# fixed cost is already paid by the time we get here (torch+triton imported)
t_fixed = t_start
print(f"process+torch+triton import already done at t={0:.1f}s (measured by caller)")

def build(compute, bm, bn, stages):
    s = BASE
    for k, v in {"ATTN_BLOCK_M": bm, "ATTN_BLOCK_N": bn, "ATTN_NUM_STAGES": stages,
                 "COMPUTE_DTYPE": f"'{compute}'",
                 "IO_DTYPE": "'fp16'" if compute in ("fp16", "bf16") else "'fp32'"}.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s); assert n == 1, k
    return s

grid = list(itertools.product(("fp16", "bf16", "tf32", "ieee"), (64, 128), (32, 64), (1, 2, 3)))
print(f"screening {len(grid)} configurations in ONE process\n")
marginal = []
results = []
for compute, bm, bn, stages in grid:
    t0 = time.perf_counter()
    p = W / f"b_{compute}_{bm}_{bn}_{stages}.py"
    p.write_text(build(compute, bm, bn, stages))
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        k = mod._attn_fwd.warmup(
            torch.empty(1, device=dev), torch.empty(1, device=dev), T, 1.0, 1, 1, 1, 1,
            NH=NH, HS=hs, HS_PAD=hs_pad, KOFF=C, VOFF=2 * C, BLOCK_M=bm, BLOCK_N=bn,
            MODE=MODE[compute], num_warps=8, num_stages=stages, grid=(1, 1))
        shared = k.metadata.shared
    except Exception as e:
        shared = None
    dt = time.perf_counter() - t0
    marginal.append(dt)
    results.append((compute, bm, bn, stages, shared))

total = time.perf_counter() - t_start
import statistics
print(f"  {len(grid)} configs screened in {total:.2f}s total")
print(f"  marginal per config: median {statistics.median(marginal)*1000:.0f} ms  "
      f"min {min(marginal)*1000:.0f}  max {max(marginal)*1000:.0f}")
over = sum(1 for *_, s in results if s and s > 101376)
print(f"  {over}/{len(results)} exceed the 101376 limit -> would be refused")
print(f"\n  compare: {len(grid)} SEPARATE worker processes at 16.7s = "
      f"{len(grid)*16.7/60:.1f} min")
print(f"  one batched process:                              {total/60:.2f} min "
      f"-> {len(grid)*16.7/total:.0f}x cheaper")
