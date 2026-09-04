# Measurement: re-tuning an identical space swings the result by up to 2.1%

The two no-op expansions fixed in `4030458` accidentally produced a controlled
repeatability experiment: the same candidate, the same source, a **byte-identical** search
space, tuned twice with 40 fresh trials each.

| candidate | pass 1 | pass 2 | swing |
|---|---|---|---|
| cand-f4a2ce82 | 3.80 ms | 3.80 ms | +0.0% |
| cand-3bcc57ce | 3.91 ms | 3.83 ms | **−2.1%** |

`min_improvement_pct` is **2.0**. So on this task a re-tune over an unchanged space
produced a swing that clears the threshold the harness uses to decide whether a change is
a real improvement.

## What this does and does not affect

Stated carefully, because n=2 and one of the two pairs showed no swing at all. This is a
bound-setting observation, not a distribution.

**It does not corrupt convergence in the dangerous direction.** `min_improvement_pct` is
consumed in `ConvergencePolicy.family_verdict`, comparing round-over-round family bests
from `FAMILY_ROUND_RECORDED`. A family's best comes from `FamilyManager.update_best`, which
is **monotonic** — it never regresses. So a noisy-low re-tune can only *lower* a recorded
best, which means:

- it can make a round look like a ≥2% improvement and buy a family rewrite rounds it did
  not earn — erring toward *more* structural search, the safe direction for a paper arguing
  against premature convergence;
- it **cannot** cause a premature freeze, because the recorded best never rises.

**It does contaminate the reported number.** The family best is what gets quoted as the
result, and a 2.1% noise band sits directly on it. Combined with the two effects already
documented — `tuned_ms` running 1.5–6.7% optimistic against `final_reeval_ms`
(`opop-v2-reeval-gap-is-the-real-number`), and the one-slow-sample asymmetry that
understates candidate speedups (`finding-one-slow-sample-per-measurement.md`) — a quoted
`tuned_ms` carries at least three independent sources of error, pointing in different
directions.

## Why the two pairs differ

`cand-f4a2ce82`'s space has 4 knobs and its optimum sat at a flat region (3.80 reproduced
exactly). `cand-3bcc57ce`'s best trials cluster at 3.83–3.91 with no clear winner, so which
configuration the TPE happens to sample decides the reported best. That is consistent with
the one-slow-sample finding: when per-trial means carry a large fixed cost, close
configurations become hard to separate, and the argmin over 40 trials is partly luck.

Not a claim about the tuner's quality — TPE is not being asked to distinguish 3.83 from
3.91 reliably, and a 40-trial budget on a noisy objective cannot.

## Not actionable yet

The obvious response — raise `min_improvement_pct` above the noise band — is wrong on this
evidence. It would make the harness *less* willing to grant rewrite rounds, and the only
demonstrated effect of the noise is granting extra ones. It would also be tuned to n=2.

The right sequence is: fix the measurement first (the one-slow-sample diagnostic, which
needs per-sample retention), then re-measure repeatability on clean numbers, then decide
whether any threshold needs to move. Recorded here so the 2.1% figure is on file when that
happens.
