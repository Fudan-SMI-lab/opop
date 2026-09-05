# Result: L3:21 rerun — 15.8 ms, the first L3:21 kernel to beat its own baseline

`run-l3-21-20260905-071312`, finished in **2.05h** of a 12h budget. Supersedes the
provisional numbers in `inprogress-l3-21-rerun-vs-0904.md`: the final re-evaluation has
run, so the verdict below is established rather than projected.

## The verdict

`cand-7dcdbd99`, `fam-f069ef3c`, origin = rewrite of `cand-1eee8139`.

```
θ_best = {PW_BLOCK_M: 64, PW_BLOCK_N: 128, PW_BLOCK_K: 16, PW_WARPS: 8, PW_STAGES: 2,
          COMPUTE_PRECISION: fp16, APPLY_BLOCK: 256, FINISH_WARPS: 1, APPLY_WARPS: 2}
```

| | ms |
|---|---|
| tuned (20 samples, quick_test) | 15.50 |
| **final re-eval (100 samples, fresh process)** | **15.80** |

Correctness on the re-eval: **5/5 trials on fresh inputs**, `correctness_mode:
dual_witness_relaxed`, `compiled: true`, `excessive_speedup: false`. Latency
`{mean 15.8, std 0.267, min 14.2, max 16.2, n 100}` — a 1.7% relative std, and all four
kernels report **0 register spills**.

### Against the baselines, at the same precision

The candidate computes in fp16, so `torch_compile_tf32` is its honest denominator.

| baseline | ms | vs 15.80 |
|---|---|---|
| eager | 25.30 | 1.601x |
| eager_tf32 | 20.90 | 1.323x |
| torch_compile | 22.20 | 1.405x |
| **torch_compile_tf32** (its denominator) | 16.30 | **1.032x** |

```json
"honest_verdict": {"candidate_precision": "fp16",
                   "compared_against": "torch_compile_tf32",
                   "same_precision_speedup": 1.0316,
                   "beats_same_precision_baseline": true}
```

**This is the first L3:21 candidate in six runs to report
`beats_same_precision_baseline: true`.** Every previous best failed it: 09-04's
`cand-05f1118a` at 0.789x, this run's `cand-1eee8139` at 1.083x (but against the *ieee*
compile path, because it is fp32), `cand-d31b0474` at 0.953x, `cand-7dcdbd99`'s own
earlier space at 0.964x.

## The re-eval gap cut the margin roughly in half

Worth stating plainly because the provisional write-up leaned on it. `tuned_ms` 15.50 →
`final_reeval_ms` 15.80 is **+1.9%**, right in line with this task's +1.5% yesterday, and
it moves the win from **5.2% to 3.2%**:

| | vs torch_compile_tf32 |
|---|---|
| on `tuned_ms` 15.50 | 1.052x |
| on `final_reeval_ms` 15.80 | **1.032x** |

The claim survives, but the margin is 3.2% against a baseline whose own std is 0.3 ms
(1.8%). So this is a real win and a **narrow** one — not the comfortable margin the
provisional number suggested. This is exactly why `opop-v2-reeval-gap-is-the-real-number`
exists.

## Progression, and where the gain came from

| candidate | ms | class | origin |
|---|---|---|---|
| cand-080f8c60 | 25.00 | scalar fp32 | seed |
| cand-1eee8139 | 20.50 | scalar fp32 | seed |
| cand-d31b0474 | 17.10 | fp16 tensor-core | rewrite of cand-1eee8139 |
| **cand-7dcdbd99** | **15.50 → 15.80** | **fp16 tensor-core** | rewrite of cand-1eee8139 |
| cand-c0b3b7cd | 25.00 | **dead branch** | rewrite of cand-080f8c60 |
| cand-fdb4dac6 | 26.00 → 25.60 | tf32→ieee, repaired | rewrite of cand-080f8c60 |

**23% better than any previous L3:21 run** (best across five prior runs: 20.50 tuned /
20.80 re-eval), and the gap to `torch_compile_tf32` closed from 26% to a 3.2% *lead*.

The mechanism is the two-loop design working as intended, and the attribution matters:

1. **The structural loop supplied the idea.** Both 17.10 and 15.50 are rewrites of the
   same 20.50 ms scalar parent under one hypothesis — move the pointwise convolution onto
   the tensor cores. Two independent kernels resulted (7830 vs 9636 bytes, 3 vs 6
   `tl.dot`), both ~18% faster than the parent.
2. **The parameter loop supplied the precision.** `COMPUTE_PRECISION=fp16` was found by
   the tuner, not written by an agent: 25 of the candidate's fp16 trials succeeded against
   a best ieee trial of 21.6 ms (`result-one-knob-decides-correctness.md`).
3. **The final 8.3% was re-tune coverage, NOT improvement K.** 16.90 → 15.50 came from a
   second 40-trial budget reaching a combination sampled **0 times** in the first 40 and
   **9 times** in the second. Every value in the winner pre-existed the expansion; the
   best trial *using* a new choice was 16.00 ms. K's two expansions this run were 0 for 2
   on their own terms.

## What the run cost, honestly

| | |
|---|---|
| wall clock | **2.05h of 12h** |
| candidates registered | 8 |
| candidates tuned | 6 |
| trials | 374 |
| **trials spent on a kernel that never ran** | **80 (21%)** |
| families | 4, all `frozen_budget` |
| **rewrite rounds used** | **2 of 6** (1 each by two families, 0 by the other two) |

The 80 wasted trials are `cand-c0b3b7cd`
(`finding-optimization-behind-a-dead-mode-branch.md`) — a rewrite whose advertised fused
kernel sat behind `if bn.training:`, which the harness's fixed train mode never selects.
Fixed in `44256cd` / `0215b70`; the fixes are driver-side so they take effect from L3:43
onward.

**The run also stopped roughly 10 hours early.** Two families whose only seeds were
destroyed by the parameterizer-revert bug had no correct candidate, and with
`max_families_active: 2` they filled both active slots at 09:16:16 and ended the loop —
freezing the winning family with 2 of its 3 rewrite rounds unused, while its trajectory
was still steep (20.50 → 15.50 inside one round). Recorded separately in
`finding-run-stops-with-budget-unused.md`. So this result came out of **one third of the
intended structural search**.

`converged` remained unreachable, and here for a stronger reason than the label logic: no
family reached a second round, so no `best_history` window could be evaluated at all. The
freeze recorded is a single *global* `budget_exhausted`; the four families are
`frozen_budget` by assignment from the loop's cleanup path rather than by four separate
`family_verdict` freezes (`finding-converged-stop-kind-is-unreachable.md`).

## The gate finding stands, on its own evidence

Four above-floor rejections this run, plus a third program replicating the tf32
fingerprint `frac_within_tol = 0.953277` to six decimals from a second lineage
(`result-one-knob-decides-correctness.md`). Two below-floor candidates were correctly
caught, repaired, and published — the control cases showing the gate discriminates rather
than merely rejecting.

Note the tension this result does *not* resolve: the run's winner is a tensor-core kernel
that the acceptance path let through, while four other tensor-core candidates were
rejected for shortfalls as small as 2.1e-3 against the reference's own two-precision
disagreement. Both facts are true; the gate question is still the user's to decide.

## Caveats

- One run, one task. `torch_compile_tf32` at 16.30 has std 0.30, and the candidate's
  15.80 has std 0.267, so a 3.2% margin is roughly 1.7 pooled standard deviations of the
  *mean* estimates — real, but not large.
- The 3.2% figure is the honest same-precision comparison. Against `torch_compile`
  (ieee) it is 1.405x, and quoting that number instead would be comparing fp16 work to
  fp32 work.
- 21% of the GPU budget measured dead code and only 2 of 6 rewrite rounds ran, so this
  result was achieved with roughly one third of the intended structural search and no
  evidence that it had converged.
