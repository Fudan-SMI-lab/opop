# Finding: the run stopped at 2.05h of 12h with 4 of 6 rewrite rounds unused

`run-l3-21-20260905-071312` produced the project's best L3:21 result
(`result-l3-21-rerun-verdict.md`) and then **terminated with 83% of its wall-clock budget
and two thirds of its rewrite budget unspent**. Not a crash — the loop's exit condition
was satisfied for a reason unrelated to either budget.

## What the numbers say

| | |
|---|---|
| wall clock | **2.05h of 12h** |
| `rewrite_rounds_per_family` | 3 |
| rounds actually used by the two productive families | **1 each** |
| rounds used by the two other families | **0 each** |
| novelty rounds | **0** |
| stop | global `budget_exhausted` |

```
fam-f069ef3c  best 15.5   rounds_used 1   <- the winning family
fam-5dfc36d7  best 25.0   rounds_used 1
fam-c2143500  best None   rounds_used 0
fam-a43b404b  best None   rounds_used 0
```

The winning family improved 20.50 → 17.10 → 15.50 **within its single round** and was
then frozen. A second round was budgeted, funded, and never run.

## The mechanism: two starved families end the loop for everyone

`_rewrite_round` returns `progressed`, and the outer loop treats `not progressed` as
"nothing left to do" (`orchestrator.py:325`):

```python
progressed = self._rewrite_round(round_no)
if not progressed:
    added = self._novelty_round(round_no)
    if not added:
        for fam in ...:            # freeze every remaining active family
            if fam.status == "active": fam.status = "frozen_budget"
        continue                   # -> global_verdict now freezes the run
```

Inside `_rewrite_round`, a family with no correct candidate is frozen **without setting
`progressed`**:

```python
if family.best is None:
    family.status = "frozen_budget"   # nothing correct in this family
    continue                          # progressed stays False
```

Now add `max_families_active: 2`. `active_families()` yields at most two families, so the
round-by-round activation order decides everything:

```
07:49:56  fam-f069ef3c  continue          <- round 1, the two productive families
08:36:52  ROUND fam-f069ef3c 15.5 round 1
08:36:52  fam-5dfc36d7  continue
09:16:16  ROUND fam-5dfc36d7 25.0 round 1
09:16:16  fam-c2143500  continue          <- FIRST activation, and the run ends here
09:16:16  fam-a43b404b  continue
```

`fam-c2143500` and `fam-a43b404b` were activated for the first time at **09:16:16 — the
same second the run finished**. Both have `best is None`, so both took the
`family.best is None` branch, `progressed` stayed False, the novelty round added nothing,
and every remaining family — including the two that had 2 rounds left and a 15.5 ms
incumbent — was frozen as `frozen_budget`.

## Why those two families were empty, and why that compounds

Their only members are `cand-6b313c39` and `cand-89fa74fe` — **both victims of the
parameterizer-revert bug** (`finding-parameterizer-reverts-the-repair.md`). Each was
rejected 4 times while every repair attempt measured a source whose fix had been silently
removed:

```
rejections: cand-6b313c39: 4    cand-89fa74fe: 4
```

So one known bug (repairs reverted) produced two empty families, and those empty families
then triggered an unrelated loop-exit path that cut short the two families that *were*
working. The cost of the revert bug is therefore larger than the 11.8 min of agent wall
previously attributed to it: it also ended the run about 10 hours early.

## What this is not

- **Not the `converged`-unreachable finding.** That one
  (`finding-converged-stop-kind-is-unreachable.md`) is about the *label* on a freeze. This
  is about freezing at all, with budget remaining, and the two are independent: here the
  productive families never reached a second round, so no `best_history` window could have
  been evaluated whatever the label logic said.
- **Not wall-clock exhaustion.** `global_verdict` checks `elapsed_hours >=
  wall_clock_hours` first and that branch did not fire; the freeze came from
  `all(f.status != "active")`.
- **Not a resume artifact.** The run ran straight through in one process.

## Why it matters more than the wasted hours

The run's improvement trajectory was still steep when it stopped. Within one round the
winning family went 20.50 → 15.50 (−24%), and the two rewrites that achieved it were both
children of the *same* parent under the *same* hypothesis — the pattern that had not yet
been exhausted. The honest reading of this run's 3.2% win over `torch_compile_tf32` is
that it was obtained from **one third of the intended structural search**, and there is no
evidence the search had converged.

Combined with the 80 trials spent on a dead kernel
(`finding-optimization-behind-a-dead-mode-branch.md`), this run spent 24% of its GPU
budget on unreachable code and left 83% of its wall clock unused.

## Not fixed — this needs a decision

The one-line change (`progressed = True` when a family is frozen for being empty, or
excluding `best is None` families from the active cap) is **not obviously right**, and
it interacts with budget allocation:

1. **Whose budget is it?** Freezing an empty family is the correct action; the bug is that
   doing so ends *other* families' rounds. A minimal fix is to make the outer loop's
   continue-condition depend on whether any family is *eligible* for another round, not on
   whether this pass happened to attempt one.
2. **`max_families_active: 2` starves late families.** Two families were never activated
   until termination. If the cap admitted a family only when it has a correct candidate,
   the two empty ones would never have occupied a slot at all.
3. **It overlaps decision 4 already awaiting the user** (seeding `best_history` so
   `converged` is reachable). Both concern how rounds are allocated and counted; changing
   one without the other risks a run that neither converges nor exhausts sensibly.

## The L3:48 contrast confirms the mechanism

The same code, same config, one day earlier, on a task whose seeds mostly survived:

| run | empty families | rounds used per family | elapsed |
|---|---|---|---|
| L3:48 09-05 | **1** | **[3, 1, 0, 3]** | 6.08h |
| L3:21 09-05 | **2** | **[0, 1, 0, 1]** | 2.05h |

On L3:48 two families used their **full 3 rounds** (`[2.09, 2.09, 2.09]` and
`[2.46, 2.09, 2.09]`), so the loop demonstrably works. The single empty family there never
blocked anything, because one empty family cannot fill both active slots.

With `max_families_active: 2`, **two** empty families can — and that is the whole
difference between a run that exhausts its rewrite budget in 6.08h and one that stops in
2.05h with 4 of 6 rounds unused. The threshold is exactly two.

That also makes the prediction testable: a run whose empty-family count reaches
`max_families_active` should terminate early, and one below it should not. n=2 so far, on
opposite sides of the threshold.

Earlier runs cannot be checked this way — `rewrite_rounds_used` is absent from their
`RUN_FINISHED` summaries (the field predates them), so `[None, None, None, None]` there
means unrecorded, not zero.

Recorded now with the evidence on disk. `scripts/audit_convergence_stop_kinds.py` reports
the stop kinds; the round-utilization numbers above come from each run's
`RUN_FINISHED.summary.families`.
