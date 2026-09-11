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

## What it would have written, on the first production ledger there has ever been

Box 2 round 0 (`run-l3-43-20260911-052630`), driving the real `conversion_verdict` + `reconcile` with
the profiles measured on the box:

| scored against | hits | misses | vacuous |
|---|---|---|---|
| H1+H3 (the winner) vs its own profile | 4 | 1 | 3 |
| H2 vs its own profile | 2 | 5 | 0 |
| **POOLED — what the code did** | **6** | **6** | **3** |

Wrong in both directions at once:

- **The winner is diluted to a coin flip.** H1+H3 declared 8 directions, 5 falsifiable, and *every
  falsifiable one was right*: `candidate_aten_bytes` down (2.54 GB → 1.62 GB), `candidate_aten_ops`
  down (15 → 10), `shared_bytes` up (17408 → 32768), `threads_launched` up (1.05M → 4.19M),
  `peak_alloc_bytes` down (2.207 GB → 2.108 GB, below the 5% materiality floor so it scores as a
  miss). It also declined 3 with reasons. That is the strongest S2d evidence either arm has produced,
  and pooled it reads 50/50.
- **H2 is charged 5 misses for another candidate's changes.** It said `unchanged` about
  `shared_bytes`, `candidate_aten_bytes`, `candidate_aten_ops`, `threads_launched` — true of *its*
  code, which was not even tuned when the round was recorded. Each became a miss because H1+H3 moved
  those dimensions.

`render_ledger` then printed both under one `## Round 0` heading, two rows per dimension, with
nothing to say which row described which code.

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
`self.ledger` from `EXPECTATIONS_RECONCILED`, but **nothing rebuilds `round_expectations`** — the
in-flight buffer. Box 2's 16 declarations, journalled in `REWRITE_PRODUCED` at 10:40:31 and the only
production declaration set S2d has, live in memory alone. Resuming would reconcile the round against
an empty buffer: `n_declared=0`, no hits, no misses. The fix's own purpose would be defeated by
applying it.

Secondary, and each sufficient on its own:

- **A resumed run's budget clock restarts** (`_elapsed_hours` reads `self.t0`, set in
  `Orchestrator.__init__`; `cmd_resume` builds a new one). Box 2 is at 5.89 h of 12; a resume makes
  its wall-clock figure incomparable with box 1's, and arm parity on the budget is the one thing the
  paired runs cannot lose.
- **`recon=0`: no wrong ledger has been written yet.** The defect's cost is entirely in the future,
  and round 2 is where it would first land. Nothing already on disk needs repairing.
- The declarations themselves are journalled *correctly* — `REWRITE_PRODUCED.payload.expectations` is
  per candidate and always was. So **the pooled entry can be re-derived correctly offline** from the
  events for the paper, which is exactly what the table above does. The measurement survives; only
  the round-2 prompt is affected.

## What this costs the experiment, stated plainly

Box 2 will very likely reach round 2 before its budget ends (round 0 took ~30 min of Loop C; 6.11 h
remain). If it does, that one prompt carries a pooled ledger. That is a real, bounded cost to S2d(c)'s
*prompt-quality* evidence, and it is the price of not destroying S2d(b)'s *declaration* evidence.
S2d(a) (declarations made before measurement) and the reconciliation arithmetic itself are unaffected
and re-derivable.

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
