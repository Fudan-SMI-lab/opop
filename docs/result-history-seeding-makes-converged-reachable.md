# Result: seeding `best_history` makes `converged` reachable on 7 of 15 families, and cuts no live search

`docs/plan-next-round-four-fixes.md` predicted that seeding the seed-phase best as
`best_history[0]` would make `stop_kind="converged"` fire for the first time. Replayed
through the **real** `ConvergencePolicy` against every family on disk that ran enough
rounds for the test to differ:

```
families examined:                        15
newly reach stop_kind='converged':         7
freeze one round EARLIER (saved GPU time): 7
families still improving when frozen:      0
```

| run | family | seed | rounds | unseeded | seeded |
|---|---|---|---|---|---|
| l3-21 0902 | fam-3dacc96b | 25.2 | [25.2, 25.2, 25.2] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0903 | fam-97b5e645 | 24.6 | [24.6, 24.6, 24.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0903 | fam-eb982220 | 24.6 | [24.6, 24.6, 24.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0904 | fam-f14a294f | 21.7 | [21.3, 20.5, 20.5] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-21 0904 | fam-b278c226 | 21.9 | [21.5, 20.7, 20.7] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0903 | fam-59e65a76 | 31.6 | [31.6, 31.6, 31.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0903 | fam-d4be7a3a | 29.3 | [29.3, 29.3, 29.3] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0903 | fam-36474be2 | 21.0 | [19.4, 19.4, 19.4] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0903 | fam-19e13c64 | 22.9 | [19.5, 19.5, 19.5] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0904 | fam-c9461c56 | 19.6 | [19.6, 19.6, 19.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0904 | fam-ff3ef34b | 20.0 | [19.5, 17.9, 17.9] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0905 | fam-92e7c576 | 22.5 | [19.6, 19.6, 19.6] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0905 | fam-4aea322a | 14.2 | [11.0, 11.0, 9.7] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-48 0905 | fam-99aee6de | 2.1 | [2.1, 2.1, 2.1] | round 4: budget_exhausted | **round 3: converged** |
| l3-48 0905 | fam-74c41d8d | 3.5 | [2.5, 2.1, 2.1] | round 4: budget_exhausted | round 4: budget_exhausted |

## The safety property, which is the part that matters

The risk with reaching `converged` sooner is deleting structural search — the failure this
project exists to avoid. It does not happen here, and the pattern in the table says why:

**Every one of the 7 newly-converged families is flat across all three rounds** — 25.2/25.2/25.2,
24.6/24.6/24.6, 31.6/31.6/31.6, 19.6/19.6/19.6, 2.1/2.1/2.1. Their seed equals their
final best; the rewrites achieved nothing measurable.

**Every family that improved keeps its full budget**, including the two that mattered most:
`fam-4aea322a` (14.2 → 11.0 → 11.0 → 9.73, the project's best L3:43 result) and
`fam-ff3ef34b` (20.0 → 19.5 → 17.9, which produced L3:43's earlier winner). Both still
freeze at round 4 on `budget_exhausted`, unchanged.

`scripts/audit_history_seeding.py` checks this explicitly and exits 1 if any family still
gaining ≥ `min_improvement_pct` is frozen earlier. It exits 0.

## A methodological correction worth recording

My first replay of this reported **0 changes** and I nearly filed the fix as inert. The
error: it evaluated each family once at `used == len(history)`, i.e. *after* its last
round, when the verdict that decides anything is taken **before** each round
(`_rewrite_round` calls `family_verdict` at the top of the loop). Walking the rounds
properly shows 7 of 15.

The wrong version is a plausible-looking snippet that produces a confident null result,
which is why the walk now lives in a script with the ordering spelled out in its
docstring rather than being re-derived each time.

## What this does and does not buy

- **A real `converged` label**, so the report can distinguish "this branch stopped paying"
  from "this branch ran out of rounds". Those were indistinguishable before
  (`finding-converged-stop-kind-is-unreachable.md`).
- **One rewrite round saved on 7 of 15 families.** At the observed 0.5–1.1h per family
  round on L3:21 that is real time, redirected to families that are still moving.
- **It does not change any reported latency.** Every family in the table reaches the same
  best either way; the seeded runs simply stop describing a flat branch as unfinished.

The second half of the fix — the slope no longer being one round stale — is not measured
here. It changes `active_families()` ordering rather than a freeze verdict, so its effect
only appears in a live run's activation sequence, which is one of the things the current
round is watched for.

Reproduce with `python scripts/audit_history_seeding.py`.
