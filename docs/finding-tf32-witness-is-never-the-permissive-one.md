# Finding: the dual-witness gate's tf32 arm has never relaxed anything — 279 of 279

Improvement A's premise is that comparing against **two** references (ieee fp32 and tf32)
is more permissive than comparing against one: a candidate that computes in tf32 should
look correct next to the tf32 reference even if it drifts from the ieee one.

Measured over every rejection on disk that carries the full dual-witness metrics:

```
n = 279 rejections, across L3:21 and L3:43, candidate dtypes fp16/bf16/tf32/none

which witness gave the HIGHER frac_within_tol?     ieee 279    tf32   0
which witness gave the HIGHER cosine?              ieee 279    tf32   0
cases where the tf32 arm alone would have PASSED the gate:      0
```

Not "rarely useful" — **never**, on either gate criterion, without a single exception.
The `or` in `worker_main.py:571` has never changed a verdict.

## Why this is arithmetic, not luck

The gate accepts `close(cand, ref_tf32) or close(cand, ref_ieee)`. Write the candidate's
output as `r + e_c` where `r` is the exact result, and the tf32 reference as `r + e_t`.

- distance to the ieee witness ≈ `|e_c|`
- distance to the tf32 witness = `|e_c - e_t|`

`e_t` is an **independent** error term. Adding an independent perturbation to a comparison
can only widen its expected spread, so the tf32 arm is the *stricter* test unless the
candidate reproduces tf32's own rounding decisions elementwise — which requires the same
accumulation order and the same tile shapes as cuBLAS, not merely the same precision.

A monte-carlo over 2M elements, sweeping the candidate's error from 0.5× to 4× the
reference's own:

```
candidate error 0.5x the tf32 ref's:  frac ieee=0.9545  tf32=0.6294
candidate error 1.0x                  frac ieee=0.6836  tf32=0.5208
candidate error 2.0x                  frac ieee=0.3825  tf32=0.3444
candidate error 4.0x                  frac ieee=0.1974  tf32=0.1920
```

The tf32 arm is worse at every error scale, and the observed data agrees quantitatively.
For independent errors of comparable size the ratio of deviations should approach
`sqrt(2) = 1.414`; measured on the tf32/ieee candidates it is **1.647** (n=44). Where the
candidate's own error dominates (fp16/bf16) the ratio should fall back toward 1 — measured
**2.254**, i.e. still worse, because bf16's error is not small relative to tf32's.

```
median_rel_err ratio (vs-tf32 / vs-ieee): median 1.647   ratio > 1 in 279/279
```

## The observation that makes it concrete

L3:21 `cand-fe183b2d` computing at **tf32**, rejected:

```
vs ieee ref: frac_within_tol 0.953277
vs tf32 ref: frac_within_tol 0.923261    <- its OWN precision, and 3% worse
```

A tf32 candidate matches the ieee reference better than it matches the tf32 reference.
That is the exact inversion of the premise the second witness was added for.

## What it costs, and what it does not

**The cost is GPU time, not correctness.** `_relaxed_close` is called twice per trial per
witness and the reference is evaluated at both precisions in `run_relaxed_correctness`
(lines 564-567): two forward passes of the reference where one would do. No verdict has
ever depended on the second one.

**It is not the cause of the rejections.** Removing the tf32 arm would change zero of the
279 outcomes. The rejections are the *floor* problem
(`finding-unreachable-correctness-gate.md`): all three tasks have noise floors below the
0.99 gate (0.9554 / 0.9767 / 0.9778). This finding says the dual-witness mechanism is not
the mitigation it was believed to be — the floor problem is untouched by it.

**It does not make the gate wrong.** Accepting on `either` is still sound; one arm is
simply inert. So there is no correctness urgency, only a dead mechanism and a wasted
reference evaluation per trial.

## No fix applied

Two options, and neither should be taken without a decision:

1. **Drop the tf32 witness.** Saves one reference forward pass per correctness trial.
   Against: it is the *cheap* half of the pair to keep, its output feeds the noise-floor
   number in every failure message (`_relaxed_metrics(out_ref_ieee, out_ref_tf32)`), which
   is load-bearing for repair diagnosis, and 279 observations are all *rejections* — the
   sample is conditioned on failure, so a candidate that genuinely matches tf32 would have
   passed and never entered this table. The premise is falsified as a *relaxation*, but the
   tf32 pass is still what measures the floor.
2. **Replace the second arm with a floor-relative test** — compare the candidate's
   deviation against the reference's own ieee-vs-tf32 deviation, which is what the tf32
   pass is already computing. This is the mechanism the user has twice deferred
   (`finding-unreachable-correctness-gate.md`), and it should stay deferred; recorded here
   only because this measurement narrows what a correct version would look like: the floor
   is already being computed on every failing trial, so the data a floor-relative gate
   needs is on hand and costs nothing extra.

Reproduce: parse `failure_detail` on every `TRIAL_DONE` with
`failure_kind == "correctness_mismatch"` and compare the `vs ieee ref` / `vs tf32 ref`
metric dicts. 279 such records exist across `runs/`.
