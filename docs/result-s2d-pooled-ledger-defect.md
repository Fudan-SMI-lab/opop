# S2d pooled every rewrite candidate's predictions into one ledger entry

**Status: FIXED in the code (`ccca414`), NOT restarted.** The fix is driver-side, so it cannot reach
the three live runs; the reasoning for letting them finish is below, and it is not "the fix is small".

## The defect

`_record_reconciliation` scored **every declaration a round produced** against **one**
`resource_deltas` map — the round's, i.e. the family incumbent's before/after.

A rewrite round produces more than one candidate. Measured across every completed L3 run:

| run | rounds with rewrites | candidates per round |
|---|---|---|
| `run-l3-21-20260908-232211` | 3 | 2, 2, 2 |
| `run-l3-48-20260907-202457` | 3 | 2, 2, 2 |
| `run-l3-48-20260909-115701` | 3 | 2, 2, 2 |

**9 of 9 rounds, 100%, exactly two candidates.** So this is the norm, not a corner. And the two are
asked for *different hypotheses* from the same analyst report, which means their declarations
routinely contradict each other **about the same dimension, correctly**, because they describe
different code.

## What it wrote, on the first production ledger there has ever been

Box 2's first `EXPECTATIONS_RECONCILED` fired at seq 717 (`run-l3-43-20260911-052630`, family
`fam-efd15aa9`, round 1) — the pooled entry, exactly as predicted, read off disk:

```
candidate_id=None      hypothesis_id='H1+H3,H2'      hits/misses/vacuous = 7 / 6 / 3
n_declared=16          16 rows for 8 dimensions -- each dimension appears TWICE
```

**The pooling has a fingerprint that needs no recomputation.** Four `rel` values each appear exactly
twice in the 16 rows, once per candidate:

| dimension | `rel`, appearing twice | what it measures |
|---|---|---|
| `candidate_aten_bytes` | 0.3616 | 2.54 GB → 1.62 GB |
| `candidate_aten_ops` | 0.3333 | 15 → 10 |
| `threads_launched` | 6.0000 | 1.05 M → 7.34 M |
| `peak_alloc_bytes` | 0.0451 | below the 5% materiality floor |

A `rel` is a property of a *measurement*. The same one appearing under two different candidates' rows
is the two candidates being scored against one measurement, visible in the log itself.

### The correct ledger, re-derived from the same events

Each candidate's own best profile from its own `TRIAL_DONE` records, the parent reading recovered from
the pooled entry's own `before` fields, and then the real `reconcile()` per candidate:

| scored against | hits | misses | vacuous |
|---|---|---|---|
| `cand-2d8eaf9a` H1+H3 (the winner, 2.8616 ms) | **3** | 2 | 3 |
| `cand-3760b4d7` H2 (3.2031 ms) | **6** | 1 | 0 |
| correct totals | **9** | **3** | 3 |
| **POOLED — what the code wrote** | 7 | **6** | 3 |

**The pooling doubled the misses, 3 → 6, and lost 2 hits.** Both candidates are misrepresented, in
opposite directions:

- **H2 is charged 5 misses it did not earn.** Pooled it reads 2 hits / 5 misses; measured against its
  own code it is **6 hits / 1 miss** — the most accurate set of predictions either arm has produced.
  It said `unchanged` about `shared_bytes`, `candidate_aten_bytes`, `candidate_aten_ops`,
  `threads_launched` and `peak_alloc_bytes`, and every one of those was **true of its own code**. Each
  became a miss only because H1+H3 moved those dimensions.
- **H1+H3, the round's winner, keeps 3 of its hits but its `occupancy` row goes from `flat` to
  `unknown`** — its own profile has no occupancy reading, so honestly reconciled that declaration is
  *unmeasured*, not a judgement. The pooled entry silently supplied the incumbent's number instead.
  This is the specific failure `_candidate_conversion` now returns `{}` for.

Note the direction: my earlier estimate of this entry, made before it existed, guessed 4/1/3 and 2/5/0
and pooled 6/6/3. The pooled figure was nearly right (7/6/3) but I had the two candidates **backwards**
— I predicted the winner would be the accurate one and it is H2. The defect is real either way and the
doubled-miss count is worse than estimated, but the per-candidate split was a guess and is now a
measurement.

`render_ledger` printed both under one `## Round 1` heading, two rows per dimension, with nothing to
say which row described which code.

## Why that mattered more than a wrong number in a log

The ledger's **only** outlet is the next round's rewriter prompt — verified: `self.ledger` is read in
exactly one place (`orchestrator.py:2388` → `RewriterInputs.ledger_entries` → `render_ledger`). S2d's
hard boundary holds, so no ranking, budget or acceptance decision was ever touched.

But that outlet is the whole treatment. An agent that adjusts to *wrong* feedback is worse off than
one with none — the measured shape of KernelPro's raw-counter arm, 1.77x against 3.35x for no
feedback at all. A round-2 prompt telling H1+H3 that its correct reasoning was half wrong is not a
cosmetic defect in the arm being tested; it is the arm being tested, inverted.

## The reconciler was already right, and said so

`reconcile`'s own caveat: "this compares the parent's best configuration against **the child's** best
configuration" — singular. The pooling was entirely in the caller. Worth recording because the module
under suspicion was the innocent one.

## Why the runs were NOT restarted

The stronger reason is not cost:

**A restart would destroy the very declarations at issue.** `_restore_family_control_state` rebuilds
`self.ledger` from `EXPECTATIONS_RECONCILED`, but **nothing rebuilt `round_expectations`** — the
in-flight buffer. Box 2's 16 declarations, journalled in `REWRITE_PRODUCED` at 10:40:31 and the only
production declaration set S2d has, live in memory alone. Resuming would reconcile the round against
an empty buffer: `n_declared=0`, no hits, no misses. The fix's own purpose would be defeated by
applying it.

> **That resume gap has since been fixed too** (`57be4df`) — declarations are now restored from
> `REWRITE_PRODUCED` for any candidate with no `EXPECTATIONS_RECONCILED` of its own, and a comment
> claiming both structures restored from the same event was simply wrong. It does **not** change the
> decision for these three runs: the fix is driver-side, so the *already-running* orchestrators still
> hold the old code and would still lose the buffer on restart. It removes the hazard for the next
> run, not for this one.

Secondary, and each sufficient on its own:

- **A resumed run's budget clock restarts** (`_elapsed_hours` reads `self.t0`, set in
  `Orchestrator.__init__`; `cmd_resume` builds a new one). Box 2 is at 5.89 h of 12; a resume makes
  its wall-clock figure incomparable with box 1's, and arm parity on the budget is the one thing the
  paired runs cannot lose.
- **`recon=0` at the time of the decision: no wrong ledger had been written yet.** That is no longer
  true — the pooled entry has since landed (seq 717, measured above) and the round-2 prompt will carry
  it. It does not change the decision, because the alternative was losing the declarations entirely,
  and the entry on disk is re-derivable: the table above IS the correct ledger, computed from the same
  events.
- The declarations themselves are journalled *correctly* — `REWRITE_PRODUCED.payload.expectations` is
  per candidate and always was. So **the pooled entry can be re-derived correctly offline** from the
  events for the paper, which is exactly what the table above does — now on the real entry rather than
  a projection. The measurement survives; only the round-2 prompt is affected.

## What this costs the experiment, stated plainly

Box 2 reached round 1 and its ledger entry is pooled, so **round 2's rewriter prompt will carry one
pooled entry** — measured, not projected. That is a real, bounded cost to S2d(c)'s *prompt-quality*
evidence: the round-2 prompt will tell H2 that 5 of its 6 correct predictions were wrong, which is the
failure mode that makes wrong feedback worse than none. It is the price of not destroying S2d(b)'s
*declaration* evidence, which a resume would have erased.

S2d(a) (declarations made before measurement) is unaffected — the 16 declarations are journalled per
candidate and correct. The reconciliation arithmetic is unaffected and, as shown above, fully
re-derivable: **for the paper, report the re-derived per-candidate table, not the journalled entry**,
and cite the `rel`-appears-twice fingerprint as the reason the journalled one is known to be pooled.

The next run started from `ccca414` gets per-candidate attribution from round 0.

## Guard

`scripts/revert_check_reconcile_attribution.py`, three variants — restoring the pool, letting an
unmeasured candidate inherit the round's deltas, and dropping the candidate from the heading. All
three CAUGHT.

Two of them read **NOT CAUGHT on Windows first**, and the reason is the recorded rule rather than a
harness bug: `_orch` calls `pytest.importorskip("optuna")`, so the test *skipped*, and a skip is not a
verdict. Both are CAUGHT on the A800, where 718 tests pass. This is the second time in this project
that a Windows skip has hidden a real failure from a revert-check — the other being
`test_environment_defect.py`'s swapped `_run_trial` arguments, which failed on the A800 for six
commits while reading green locally.
