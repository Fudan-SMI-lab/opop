# Prediction for `run-l1-19-20260906-220044` (Loop C smoke), recorded before the evidence

Written at launch so the outcome can falsify it. Config: `configs/smoke_l1_loopc.yaml`
(2 seeds, `rewrite_rounds_per_family: 2`, `no_improve_rounds: 1`,
`min_improvement_pct: 50.0`, `max_families_active: 2`, `max_families_total: 4`).

## Why this run exists

The D1-D4 verification run (`run-l1-19-20260906-202221`) used
`rewrite_rounds_per_family: 0`, so it recorded `FAMILY_ROUND_RECORDED=0`,
`REWRITE_PRODUCED=0`, and exercised only generator/parameterizer/analyst/novelty. Three
things the fixes touch were therefore verified only on the no-rewrite path:

1. **`_rewrite_round`'s loop body** — the rewrite branch, `record_round`,
   `FAMILY_ROUND_RECORDED`. With rounds=0, `family_verdict` freezes before reaching it.
2. **`idle_rounds = 0`** (orchestrator.py:581), the guard's reset after a productive round.
   Never executed, so a *false* `OUTER_LOOP_STUCK` on a healthy run is still unfalsified —
   and that would end a run early, which is the exact class of bug D2 was.
3. **`stop_kind = "converged"`** — 36 of 36 recorded family freezes are `budget_exhausted`;
   `converged` has never fired. `FAMILY_SEEDED` seeds `best_history` to fix the off-by-one,
   but nothing has tested it.

## Predictions

| # | prediction | what falsifies it |
|---|---|---|
| P1 | `FAMILY_ROUND_RECORDED` >= 1 and `REWRITE_PRODUCED` >= 1 | 0 of either: Loop C still does not run, and the fixes remain untested on this path |
| P2 | `OUTER_LOOP_STUCK` = 0 | any occurrence: the guard fires on a healthy run, i.e. `idle_rounds = 0` does not reset — a **new early-stop defect of my own making** |
| P3 | at least one `CONVERGENCE_DECIDED(scope=family, stop_kind="converged")` | only `budget_exhausted`: the off-by-one is NOT fixed and `FAMILY_SEEDED`'s docstring overclaims |
| P4 | `rewriter` appears in `AGENT_CALL_FINISHED` modules | absent: the rewriter agent is still unexercised since the fixes |
| P5 | the run ends via `global freeze`, total events < 1000 | a spin, or a kill |

P3 is the one I am least sure of. `min_improvement_pct: 50.0` makes the improvement test
*fail* by construction on L1:19, which is what should produce `converged` — but if the
budget check still runs first, or if `best_history` is not actually seeded before Loop C,
it will report `budget_exhausted` again. Either outcome is informative; only P3 failing
while P1 passes tells me the seeding fix is inert.

P2 is the one that matters most if it fails: it would mean I shipped a second
early-termination defect today, in the guard added to prevent the first.

## Not predicted

Whether the rewrites make L1:19 faster. Irrelevant here — the run exists to execute
branches, not to produce a result. `trials_per_space: 4` is far too small for a real
number.
