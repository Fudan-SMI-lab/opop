# Result: the fp64 relative gate fired on its first run, and it is load-bearing

`configs/experiments_l3.yaml` turned `fp64_relative_gate` on as the next round's test
configuration, with the falsification rule that a rescued candidate failing its final
re-eval means the multiplier is too loose. First reading, 0.9h into
`run-l3-21-20260905-195615`:

```
correctness jobs with the gate armed : 195
jobs where the relative arm decided  :  38  (19.5%)
correctness trials rescued           : 129   (93 tf32, 36 fp16)
jobs where the fp64 golden was unavailable : 0
ratio_to_reference: n=38  min 0.882  max 0.920
rescues that used the multiplier's slack (ratio > 1.0): 0
```

Read `python scripts/audit_fp64_gate.py`. The run started 19:56 and the driver-side
journalling landed 20:44, so this run carries the data only in `jobs/*.out.json`; the audit's
fallback covers exactly that case.

## The finding: every rescue is a candidate MORE accurate than the reference

Not one of the 38 rescues used the slack the multiplier exists to grant. Every ratio is
**below 1.0**, meaning the candidate sits closer to the fp64 truth than the tf32 reference it
is being compared against. The gate is not loosening the standard here; it is removing a
comparison against a noisy yardstick.

That is the mechanism `finding-tf32-witness-is-never-the-permissive-one.md` and
`opop-v2-all-three-tasks-floors-below-gate` predicted, now observed in the direction they
predicted: 93 of the 129 rescued trials are **tf32** candidates, the class that could not pass
before.

## It decided the run's fourth seed

`cand-61759130`'s best trial `tr-fdefba48` (22.8 ms, `COMPUTE_DTYPE=tf32`) is a rescued
trial. Without the arm that candidate is rejected and the seed contributes nothing. The other
four space-bests are clean, so the gate is currently load-bearing on one of five.

Verified independently in a fresh process, 5 trials, `scripts/verify_rescued_trial.py`:

```
trial 0..4:  ABSOLUTE gate -> FAIL   (0/5 pass)
   vs tf32 ref : frac=0.9279   cosine=0.999999352
   vs ieee ref : frac=0.9589   cosine=0.999999791
   reference's OWN ieee-vs-tf32 floor : frac=0.9553
   RMSE vs fp64: ieee-ref 7.05e-07   tf32-ref 7.03e-04   cand 6.47e-04
   candidate / tf32-reference error ratio = 0.9200
ratio over 5 trials: min 0.9196  max 0.9204  mean 0.9199
more accurate than the reference: 5/5
smallest multiplier that would admit every trial: 0.9204
```

Three things in that block matter more than the verdict:

1. **The candidate would pass at multiplier 1.0.** The configured 2.0 is not what admitted
   it; being genuinely more accurate is. So this rescue cannot be blamed on a loose constant,
   and the falsification rule is not triggered by it.
2. **`frac_within_tol` = 0.9279 against tf32 while the reference's own floor is 0.9553.** The
   candidate is *inside* the band the reference itself occupies. The absolute gate demands
   0.99 of both, which neither meets — it was rejecting the task, not the kernel.
3. **cosine is 0.99999935**, far above its 0.99985 threshold, on a rejected trial. The gate is
   effectively single-criterion on `frac`, exactly as
   `research-tolerance-practice-under-reference-noise.md` measured (cosine passes 279/279).

## What would falsify it, and has not

- **A rescue with ratio near or above the multiplier.** None: the worst is 0.920.
- **A rescued candidate failing the final 5-trial re-eval.** Not yet reachable — the run's
  final re-eval is hours away. `audit_fp64_gate.py` checks it automatically and prints the
  turn-it-off instruction if it happens.
- **The gate rescuing something on a task with no low-precision knob.** Consistent with the
  measured opportunity distribution: L3:48 has 0 of 12 spaces with a precision knob and will
  never trigger it, so a rescue there would mean the detector is wrong.

## Correction to a claim I made earlier

I described L3:43 as "the experiment's main test" because 26/26 of its spaces carry a
precision knob, and L3:21 as a weaker one at 6/10. L3:21 has now produced 38 rescues before
its first rewrite round, so the arm is exercised well here too. The 26/26 figure predicted
*opportunity*, and I let it stand in for *expected firing rate* without saying so.

Reproduce with `python scripts/audit_fp64_gate.py`, and re-verify any single rescue with
`python scripts/verify_rescued_trial.py <trial.py> <reference.py>` inside the WSL venv.
