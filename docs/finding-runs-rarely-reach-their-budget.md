# How runs actually end: 1 of 19 was ended by the wall clock

**Status: measurement, 2026-09-06.** Produced by `scripts/audit_run_termination_reasons.py`
over the 19 real runs plus 15 smokes/tunefiles on disk. Reads only.

Written after D4 (an outer loop that could not terminate — see
`finding-loop-d-preflight-defects.md`) to answer the general form of the question: is
`RUN_FINISHED` reached because the budget was spent, or because some freeze rule fired early?

## The distribution

| ending | runs |
|---|---|
| `wall_clock` — clock actually spent | **1** (`run-l3-21-20260905-195615`, 12.82 h / 106.8%) |
| `nothing_active` — every family frozen, clock left | **13** |
| `killed` — no `RUN_FINISHED` | 5 (my interventions, plus the D4 spin) |

Of the 13, the median used **51.7%** of its wall clock and the worst used **10.1%**.

## The four runs that froze with no freeze verdict at all

Four `nothing_active` runs recorded **zero** `CONVERGENCE_DECIDED(scope=family,
verdict=freeze)` events, yet ended with every family `frozen_budget`. Nothing in the family
bookkeeping froze them; only the outer loop's blanket sweep could have. The rewrite rounds
they left on the table:

```
run                        rewrite rounds used   novelty calls   clock used
run-l3-21-20260903-210650              0 of 12               0       16.5%
run-l3-21-20260905-071312              2 of 12               0       17.1%
run-l3-43-20260902-140823              0 of 12               0       56.7%
run-l3-43-20260902-213608              0 of 12               0       10.1%
```

**0, 2, 0, 0 of 12.** These runs did not exhaust anything. They were ended by the sweep that
D2 removed — this is that defect's historical cost, and it is larger than the D2 write-up
estimated, because that write-up reasoned about Loop D runs while this shows the sweep also
ended runs where Loop D never fired (`novelty_calls=0` in all four: the interlock kept the
gate shut, so `added=False` was reached at the *interlock*, and the sweep then fired anyway).

The D2 write-up said this was "harmless today, because `added` is always False at the
interlock *before* anything is frozen that mattered." That was wrong on this evidence. The
sweep fired at the interlock and froze four families holding 12 unspent rewrite rounds, four
separate times.

## What is now different

* The sweep is gone (D2), so `added=False` no longer freezes anything.
* Families that cannot be rewritten are frozen where that predicate lives
  (`_freeze_unrewritable_families`, D4) rather than as a side effect of the sweep, and emit
  `FAMILY_FROZEN_UNREWRITABLE` so this audit can see them.
* `OUTER_LOOP_STUCK` bounds the failure mode the D2 fix introduced.

## What this does NOT establish

That the remaining 13 early endings were all wrong. `nothing_active` is a legitimate ending
when every family genuinely converged or spent its rounds — nine of the 13 do carry
`budget_exhausted` family verdicts. Whether *those* caps (`rewrite_rounds_per_family: 3`,
`no_improve_rounds: 2`) are set too tight for L3 is a separate question about budget policy,
not a defect, and it needs the post-fix runs before it can be answered: until today every run
was subject to the sweep, so no run on disk is clean evidence about the caps.

The one number that is now clean: **a 12 h L3 budget affords ~18 rewrite rounds and the
configs allow 20**, so wall clock and rewrite budget are roughly matched. A run ending at 17%
of its clock with 0 of 12 rounds used was not a budget-policy question.

## Re-run

```
uv run python scripts/audit_run_termination_reasons.py
uv run python scripts/audit_run_termination_reasons.py runs   # same, explicit
```
