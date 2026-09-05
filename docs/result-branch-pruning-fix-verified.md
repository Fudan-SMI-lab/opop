# Result: the branch-pruning fix worked — 4 of 4 families explored, up from 2 of 4

`81cd562` ("stop pruning search branches by incumbent latency", 09-04 23:30) replaced
`active_families()`'s latency ranking with unproven-first + improvement-slope. It has now
run on three L3 experiments and the effect is visible in the event log rather than argued
from the code.

This started as an investigation of a *suspected* defect — most L3 families across the
project never received a rewrite round — and ended as a validation. The starvation is real
but almost entirely pre-fix, and two of the three statistics I computed on the way did not
survive being dated against the fix commit. Recording all of it, since the discarded
numbers are the kind that look alarming and mean nothing.

Counting families that were never selected AND had a correct result to rewrite (the
genuinely starved set — a family whose candidates never produced a correct result is
declined on purpose):

| | families | genuinely starved |
|---|---|---|
| pre-fix (15 runs) | 50 | **12** |
| post-fix (4 runs) | 16 | **1** |

The single post-fix case is `l3-48-0905-003307`, a run that died at 0.54h having never
reached Loop C at all — no rewrite round was handed to *any* family there, so it is an
abort, not a ranking failure.

## Rounds per family, by run

```
run                      wall    families  zero-round  rounds per family
l3-21-0902-113144                    4         4       [0, 0, 0, 0]     <- pre-fix
l3-21-0903-071357                    4         2       [3, 3, 0, 0]     <- pre-fix
l3-21-0903-210650                    4         4       [0, 0, 0, 0]     <- pre-fix
l3-21-0904-013056        8.10h        4         2       [3, 3, 0, 0]     <- pre-fix
l3-43-0902-140823                    4         4       [0, 0, 0, 0]     <- pre-fix
l3-43-0902-213608                    4         4       [0, 0, 0, 0]     <- pre-fix
l3-43-0903-020233                    4         2       [3, 3, 0, 0]     <- pre-fix
l3-43-0903-145357                    4         2       [3, 3, 0, 0]     <- pre-fix
l3-43-0904-093730       11.67h        4         2       [3, 3, 0, 0]     <- pre-fix
--- 81cd562 lands 09-04 23:30 ---------------------------------------------------
l3-48-0905-010737        6.08h        4         1       [3, 3, 1, 0]
l3-21-0905-071312        2.06h        4         2       [1, 1, 0, 0]     (stopped at 2.06h)
l3-43-0905-091705       (running)     4         0       [3, 2, 1, 1]
```

`[3, 3, 0, 0]` in **6 of 6** completed pre-fix runs that got as far as rewriting at all
(the `[0, 0, 0, 0]` rows above are runs that ended before Loop C, or whose candidates never
produced a correct result): the exact same split every time, which is what a deterministic
ranking bug looks like. Two families
per L3 run were registered, parameterized, tuned — and then never once handed to the
rewriter across 8 to 11.7 hours.

## The old mechanism, confirmed on disk rather than inferred

On `run-l3-43-20260904-093730`, `fam-f7e22112` (21.2 ms) and `fam-91e005ff` (23.7 ms) both
existed from **+0.21h**, both were tuned successfully (+2.98h and +1.14h), and both have a
valid `best`. Neither appears in a **single** `CONVERGENCE_DECIDED` event for the entire
11.67-hour run — they were never selected, so they were never even *considered* for a
freeze verdict. The two ranked families alternated instead:

```
+ 4.09h  fam-c9461c56 continue   -> round 1
+ 6.66h  fam-ff3ef34b continue   -> round 1
+ 8.33h  fam-ff3ef34b continue   -> round 2
+ 8.75h  fam-c9461c56 continue   -> round 2
+ 9.97h  fam-ff3ef34b continue   -> round 3
+11.66h  fam-c9461c56 continue   -> round 3   <- 11.66h of 12h, budget gone
+11.66h  both freeze budget_exhausted
```

Six rounds, two families, and the run ends. The other two branches are reported as
`frozen_budget`, which reads as "explored and exhausted".

## The new mechanism, on the live run

`run-l3-43-20260905-091705` hands every family a first round before any family gets a
second:

```
+1.72h  fam-4aea322a used=0  -> round 1 (14.2 -> 11.0)
+2.23h  fam-92e7c576 used=0  -> round 1 (22.5 -> 19.6)
+2.44h  fam-7f682a54 used=0  -> round 1 (23.4 -> 19.9)
+3.36h  fam-ea7bc8bb used=0  -> round 1 (28.0 -> 21.3)
+4.03h  fam-4aea322a used=1  -> round 2
+4.76h  fam-92e7c576 used=1  -> round 2
+5.43h  fam-4aea322a used=2  -> round 3 (11.0 -> 9.73)
```

Rule 1 (unproven-first) is doing exactly what it was written to do, and rule 2
(improvement slope) then correctly re-funds `fam-4aea322a` — the family with the steepest
slope — for rounds 2 and 3, which is where the 9.73 ms came from.

## What this does NOT show

- **Not that the fix caused the 9.73 ms.** `fam-4aea322a` was the *best-ranked* family by
  latency too (11.0 after round 1), so the old policy would also have funded it. The fix
  bought coverage of the other two branches, not this result. Claiming otherwise would be
  the same error the pre-fix ranking made — reading a win backwards into whatever policy
  happened to be running.
- **Not that the starved families would have won.** Their seeds were 21.2/23.7 (0904) and
  25.0/31.6 (0921) against winners at 17.9 and 20.5. What was lost is the *information* of
  whether a rewrite could move them, which is precisely what `_improvement_pct` needs and
  what `result-second-ranked-family-catches-the-leader.md` shows is not predictable from
  the seed.
- **Not fully verified at 4-for-4.** `l3-21-0905-071312` still shows `[1, 1, 0, 0]`, but it
  ran only 2.06h — it stopped early for the empty-family reason in
  `finding-run-stops-with-budget-unused.md`, a different defect, and never reached a
  second selection pass. `l3-48`'s remaining `0` is one family whose candidates never
  produced a correct result (`best is None`), which the loop correctly declines to rewrite.

## The statistic I computed and then discarded

"24.5% of family-selection slots (24 of 98) went to a family that immediately froze" — and
in 8 batches a freezing family took a slot while another *active* family was waiting. Both
true, both nearly meaningless: every one of those batches is the **final** batch of its
run, where the two exhausted families are re-selected once more, return `freeze`, and the
loop exits. The slot costs a `family_verdict()` call, not a rewrite round. The genuine cost
was always the ranking, not the freeze accounting, and separating the two required dating
each run against `81cd562` rather than pooling them.

Reproduce with `python scripts/audit_family_round_coverage.py`.
