# Result: L3:21 beats its strongest baseline at 15.6 ms, and the win is clean

`run-l3-21-20260905-195615`, 1.5h into a 12h budget. First rewrite round of `fam-a4a8353c`
produced `cand-80bf3097`, which tuned to **15.6 ms** against this run's own baselines:

```
eager                25.3 ms
torch_compile        22.3 ms
eager_tf32           20.9 ms
torch_compile_tf32   16.4 ms   <- the strongest baseline
cand-80bf3097        15.6 ms   (std 0.4, min 14.7, max 16.2, n=20)   1.05x
```

Progression through the run: seeds 19.4 / 20.4 / 19.4 / 22.8, then the first rewrite → 15.6.
So the structural rewrite, not the tuner, produced the win — which is the mechanism the paper
argues for.

## It does not depend on the fp64 gate

Important for how this gets reported. `tr-b18a9d20` passed the **absolute** gate 5/5 on its own:

```
absolute gate passed 5/5 trials
   vs tf32 ref : frac=0.999688  cosine=1.000000000
   vs ieee ref : frac=0.955401  cosine=0.999999753
```

`scripts/verify_rescued_trial.py` on the materialized trial confirms `rescued=0` for its job.
The two rescued space-bests in this run are `cand-61759130`'s (22.8 ms), which is not the
headline. So the 15.6 ms number stands with or without `fp64_relative_gate`, and the gate's
contribution to this run remains "admitted a fourth seed", not "produced the best result".

## The suspicious coincidence, and what it turned out to mean

The candidate's RMSE against an fp64 golden matched the tf32 reference's to five significant
figures — 7.029395e-04 vs 7.029335e-04, ratio **1.0000**, cosine 1.000000000 — on a trial whose
PARAMS say `COMPUTE_DTYPE='fp16'`. Identical error from a different arithmetic looked like an
inert knob, i.e. the dead-branch failure of `opop-v2-dead-mode-branch-strands-optimization`.

It is not. Materializing the same candidate at all four dtype values
(`scripts/audit_dtype_knob_is_live.py`):

```
COMPUTE_DTYPE=fp16   RMSE 7.029395e-04   ratio to tf32-ref 1.0000
COMPUTE_DTYPE=bf16   RMSE 5.618008e-03   ratio to tf32-ref 7.9922
COMPUTE_DTYPE=tf32   RMSE 7.360844e-04   ratio to tf32-ref 1.0472
COMPUTE_DTYPE=ieee   RMSE 7.094173e-07   ratio to tf32-ref 0.0010

pairwise: no two variants are bit-identical (fp16 vs tf32 rmse 1.21e-03)
```

All four differ, and they order exactly as the formats predict. The match is not a coincidence
needing explanation — it is the **expected** result: fp16 and tf32 both carry a **10-bit
mantissa**, and this kernel accumulates in fp32 either way, so their dot error should agree. What
differs is exponent range (5-bit vs 8-bit), which only bites above ~65504; this output's absmax
is 5.56.

Two things worth keeping from that table:

1. **fp16 is very slightly better than tf32 on the same candidate** (1.0000 vs 1.0472), which
   corroborates `opop-v2-witness-default-only-weakness`'s measurement from the other direction.
2. **bf16 lands at 7.9922×** — 7-bit mantissa with an 8-bit exponent. The fp64 gate rejected the
   bf16 arm of a *different* candidate (`cand-fe183b2d`) at ratio **8.000** earlier today. Two
   unrelated candidates, same task, agreeing to three digits: the gate's bf16 rejection is a
   property of the format on this workload, not a fluke of one kernel. That is the clearest
   evidence so far that the gate discriminates rather than loosens.

## What is not yet settled

- **The final re-eval.** 15.6 ms is `tuned_ms` from a 20-sample quick test.
  `opop-v2-reeval-gap-is-the-real-number` measures `tuned_ms` as systematically optimistic by
  1.5–6.7%, so the reportable figure is `final_reeval_ms` at the end of the run. At the worst
  observed gap 15.6 becomes ~16.6, which would **lose** to 16.4. So this win is not safe yet and
  must not be claimed until the re-eval lands.
- **`cand-f66890d0`** (the H2 sibling, fusing BN apply with the next conv load) has not tuned yet.
- **`converged`**, still not observed; the family is one round in.
