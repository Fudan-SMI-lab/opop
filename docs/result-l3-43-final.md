# L3:43 final result — run-l3-43-20260908-053708

Terminated deliberately at 11.73 h of a 12 h budget, on request, to land the P1–P3 fixes before
spending another 24 h on L3:21 and L3:48. The run was healthy at termination (zero agent failures,
zero classify failures, all convergence verdicts `continue`) — this is a stop, not a crash, and the
result below is therefore a **lower bound** on what the budget would have produced.

## The number

θ_best was re-evaluated **independently after termination**, in a fresh process, through the
harness's own job builder with the run's own config values (5 correctness trials + 100 timed
samples, exclusive GPU, `dual_witness_relaxed` + fp64 relative gate). Not a hand-rolled timing loop.

| | ms |
|---|---|
| `tuned_ms` (20-sample median, during tuning) | 3.2558 |
| **`final_reeval_ms` (100-sample median)** | **3.0126** |

The re-eval came out **7.47% FASTER** than tuned — the second counterexample to "tuned is
optimistic" (L1:42 was the first, at 2.65% faster). The rule remains "quote `final_reeval_ms`", not
"discount tuned by ~N%".

Correctness: **5/5 trials passed**, `fp64_rescued_trials: 0` (it cleared the absolute gate outright,
without needing the fp64 relative arm), `excessive_speedup: False`.

## Speedups, on `final_reeval_ms`

| baseline | median ms | speedup |
|---|---|---|
| eager | 21.451 | 7.121x |
| eager_tf32 | 18.241 | 6.055x |
| torch_compile | 13.825 | 4.589x |
| **torch_compile_tf32** (same-precision headline) | 10.986 | **3.647x** |

The headline to quote is **3.647x against `torch_compile_tf32`**, the strongest baseline. Note the
candidate is fp16 while that baseline is tf32, so it is not strictly same-precision; the honest
framing is that it beats the best available torch baseline by 3.65x while passing a dual-precision
relaxed gate with an fp64 relative check.

## θ_best

```python
PARAMS = {
    "ATTN_BLOCK_M": 128, "ATTN_BLOCK_N": 64, "ATTN_NUM_WARPS": 8, "ATTN_NUM_STAGES": 2,
    "COMPUTE_DTYPE": "fp16", "IO_DTYPE": "fp16",
    "LINEAR_BLOCK_M": 128, "LINEAR_BLOCK_N": 128, "LINEAR_BLOCK_K": 32,
    "LINEAR_NUM_WARPS": 8, "LINEAR_NUM_STAGES": 2,
}
```

Candidate `cand-fbd1988e`, family `fam-f94ad85e`, space `sp-db9f8108`, trial `tr-bc111c61`.

Four of eleven knobs sit at a domain edge (`ATTN_BLOCK_M` high, both `NUM_WARPS` high,
`COMPUTE_DTYPE`/`IO_DTYPE` at the fp16 end), so the space still had somewhere to go — consistent
with the run being stopped rather than converged.

Compiled kernels at θ_best (from the re-eval's own Triton metadata):

| kernel | regs | spills | shared |
|---|---|---|---|
| `_attn_fwd` | 218 | 0 | 65536 |
| `_linear_fwd` | 122 | 0 | 32768 |
| `_linear_fwd` | 121 | 0 | 16384 |

**Zero spills at θ_best**, against 52 spills for the parent it was rewritten from — worth noting
beside the `rho = -0.516` spill/speed inversion recorded in `next-round-fix-plan.md` P5: the fastest
configuration of the fastest candidate spills nothing. That weakens the inversion as a general
claim and is another reason not to change the spill guidance on one task's ranking.

## Lineage — four generations, every one an improvement

```
cand-c8830fe8 (seed)     8.108 ms
  -> cand-7760976a (rewrite)  4.620 ms   -43.0%
    -> cand-80fea541 (rewrite)  3.290 ms   -28.8%
      -> cand-fbd1988e (rewrite)  3.256 ms tuned / 3.013 ms re-eval   -1.0% tuned
```

The seed was `Minimal-kernel-count design`: cuBLAS `F.linear` for the projections around one fused
Triton flash-attention kernel. The final rewrite replaced a dual-Q-tile kernel (255 regs, 52 spills,
`BLOCK_N` capped at 32) with a single-accumulator flash kernel at `BLOCK_M=128`.

## Run accounting

| | |
|---|---|
| wall clock | 11.73 h of 12 h (terminated) |
| events | 1485 |
| trials | 1004 complete-or-failed (795 ok, 181 shared-memory-doomed, 5 correctness mismatch) |
| Loop C rounds | 7 recorded, 15 rewrites, 0 evaluated-nothing |
| Loop D | **0** — 4 seed families against `max_families_total = 3` |
| families | 4, none frozen; all convergence verdicts `continue` |
| agent failures / rescues / session resets | 0 / 0 / 0 |
| median coverage | 816/816 trials, 4/4 baselines |
| time inside agent calls | 6.05 h (55%): analyst 2.61, parameterizer 1.68, rewriter 1.57, generator 0.19 |
| `impossible_fraction` verdicts | 13 of 25 (the missing fp16 ceiling) |
| verdict kinds | `compute_bound` 16, `resource_limited` 10 (no `launch_bound`, `overhead_floor`, `latency_bound`, `mixed`, `unknown`) |

## What this result is NOT evidence for

- **Not a converged optimum.** Stopped by request with the budget nearly spent but four families
  active and four knobs at domain edges.
- **Not evidence about backend choice.** All 19 candidates are Triton, because the contract says to
  prefer Triton and never mentions CUTLASS. See `backend-coverage-and-defects.md`.
- **Not a clean reading of remaining compute headroom.** θ_best is fp16 scored against a tf32
  ceiling, so its `pct_of_compute_peak` of ~141% is uninterpretable. The verdict KIND
  (`compute_bound`) is independently corroborated by arithmetic intensity 990 against a ridge of 60.
- **Not a full test of the four loops.** Loop D never ran.

## A third cause behind P2, found while doing this re-eval

`measure_launch_overhead` was passed as `True` and `launch_overhead` still came back `None`. Cause:
the overhead block exists **only in `run_eval`** (`worker_main.py:1370`), and both L3 configs set
`correctness_mode: dual_witness_relaxed`, which routes to `run_relaxed_correctness`
(lines 1643–2028) — a handler that mentions neither `measure_launch_overhead` nor `launch_overhead`
anywhere.

So P2 has three independent causes, not two: (a) only `full_eval` requests it, (b) the relaxed
handler ignores the request entirely, and (c) `classify()` divides by the wrong `gpu_ms`. Fixing
(a) and (c) without (b) would produce exactly this result — a `True` flag and a `None` value — on
the configs the experiments actually use.
