# Result: on L3:21's best candidate, one knob decides correctness and the rest decide speed

`run-l3-21-20260905-071312`, `cand-d31b0474`, 55 trials. The cleanest demonstration in the
project of the two-loop premise — that tuning feedback carries structural information the
outer loop can act on — and it arrived without being looked for.

## Outcome is determined entirely by `COMPUTE_PRECISION`

| precision | ok | correctness_mismatch | runtime_error | best ms | ieee frac when it failed |
|---|---|---|---|---|---|
| **fp16** | **25** | 0 | 1 | **17.10** | — |
| ieee | 8 | 0 | 0 | 21.60 | — |
| bf16 | 0 | **13** | 2 | — | 0.674026 – 0.67403 |
| tf32 | 0 | **6** | 0 | — | 0.953274 – 0.953277 |

Zero mismatches at fp16 or ieee. Zero successes at bf16 or tf32. **No block or warp combination
changes that** — and the space has ten other knobs spanning `PW_BLOCK_M/N/K` ∈ {16…512},
`PW_WARPS` ∈ {1…32}, `PW_STAGES` ∈ {1…5}, three more warp counts and two block sizes.

So on this candidate the block/warp knobs govern **speed** and the precision knob alone governs
**correctness**. That is exactly the separation `TuningStatsAnalyzer.failure_clusters` exists to
detect, showing up unprompted in production.

## Two more fingerprints, and one of them locates a residual bug

Both failing branches reproduce their `frac_within_tol` to ~6 decimals across independent
configurations — 13 for bf16, 6 for tf32 — which is the same signature established in
`finding-unreachable-correctness-gate.md`: the value is a property of the arithmetic, not of the
block sizes.

- **bf16 at 0.674** is far below the task's 0.955360 floor. Genuinely wrong; the gate is right
  to reject it. bf16 has 8 mantissa bits against fp16's 10, and on an O(1)-magnitude task like
  L3:21 the extra range buys nothing while the lost mantissa costs accuracy — the opposite of
  L3:48, where range is everything.
- **tf32 at 0.953274–0.953277** is 2.1e-3 *below* the floor — and it is **the same value the
  a=0 rejection reported** (0.953277) before the repair.

That second one is the useful part. The repair diagnosed a real mixed-precision defect ("the
custom pointwise convolutions forced Triton TF32 while the rest of the MBConv remained on
PyTorch's harness-controlled convolution path") and fixed it for the **ieee and fp16** branches
— both now pass — but the **tf32 branch still carries the original bug**, at the original
frac, to six decimals.

The candidate was published anyway, because publication tests the default (`ieee`) and minimal
(`fp16`) witnesses, and both are on the repaired side. The tuner then routed around the broken
branch on its own: 25 fp16 wins, 0 tf32 wins.

## Why this is a good outcome, not a hole

It would be easy to read "published with a broken branch" as an escape. It is not:

- every trial re-runs correctness before timing, so the 6 tf32 trials **failed** and contributed
  no latency;
- TPE learns from those failures and stops sampling the branch;
- the reported best (17.10 ms at fp16) is a configuration that passed correctness on every one
  of its 25 trials.

The system degraded gracefully: a partial repair plus a tuner that avoids what does not work
still produced the run's best result. That is the intended behaviour of a search that treats
failures as information rather than as errors.

## What it says for the analyst and the next rewrite round

`failure_rate_by_value` on `COMPUTE_PRECISION` is 100% for bf16 and tf32 and 0% for fp16 and
ieee — an unambiguous signal the analyst can hand to the rewriter, and a much stronger one than
any latency curve. The right structural hypothesis is not "try other block sizes" but "the tf32
path still mixes conventions; fix it and the candidate gains a second working tensor-core
precision".

**The analyst did say it, and better than I did.** From `BOTTLENECK_REPORTED` at 08:07, quoted
verbatim:

> "No parameter meets the requested definition of blocked headroom… At the 17.1 ms best
> configuration, usage is balanced well below hardware limits: 78/255 registers per thread
> (30.6%), 32,768/101,376 B opt-in shared memory, 256/1024 threads, and zero spills… **Precision
> is decisive: fp16 has a 19.15 ms median versus 39.25 ms for IEEE**, confirming an
> arithmetic-throughput/tensor-core wall in the IEEE path. The tuned winner already solves it by
> casting inputs to fp16 and accumulating `tl.dot` into fp32. **Both tf32 and bf16 failed every
> tested correctness case, so tf32 should not replace the accepted fp16 path without first
> resolving that discrepancy.** The source PARAMS still defaults to `COMPUTE_PRECISION='ieee'`,
> so the tuned fp16 configuration must be promoted before considering further structural work.
> The cluster of fp16 results at 17.1–17.6 ms across substantially different register/shared
> memory usage indicates a whole-pipeline floor…"

Every element is there and three go beyond what I derived from the same data:

1. It correctly **declined** to report blocked headroom at the three boundary knobs K expanded,
   having checked that no trial past them failed on registers, shared memory, threads, OOM, or
   compilation. That is the honest answer, and it is the answer that avoids a wasted rewrite.
2. It quantified the precision effect as a **median** (19.15 vs 39.25 ms), not just a best-case
   difference — a stronger statistic than my per-branch minima.
3. It noticed something I missed: **`PARAMS` still defaults to `ieee`**, so the tuned fp16
   configuration is not the source's default and must be promoted or the next rewrite inherits
   the slow path.

It also independently reached the tf32 conclusion — do not adopt that branch until the
discrepancy is resolved — from the failure clustering alone, without access to the a=0 rejection
message that shows the frac is unchanged from before the repair.

This is the two-loop premise working end to end: deterministic tuning statistics
(`failure_rate_by_value` 100% for bf16/tf32, 0% for fp16/ieee) became a structural
recommendation that the harness could not have written itself.

## Caveats

`tuned_ms`, 20 samples per trial, one candidate, one run. The precision separation is 55 trials
and unambiguous; the 17.10 ms figure carries the usual optimism against `final_reeval_ms` and
has not been re-evaluated yet. And per `inprogress-l3-21-rerun-vs-0904.md`, 17.10 ms at fp16 is
judged against `torch_compile_tf32` (16.30) and reports **0.953x** — it does not beat its
same-precision baseline.
