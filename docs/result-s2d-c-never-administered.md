# S2d(c) has never been administered: the ledger is per family, and no family gets a second round

**Status: MEASURED. Not a code defect — every component behaves as designed.** It is a finding about
the experiment, and it decides whether S2d(c) is testable by a 12 h run at all. It is not.

## The claim and the mechanism

S2d(c) is the claim that reconciled predictions improve the **next** rewrite. The ledger reaches a
prompt through exactly one line (`orchestrator.py`):

```python
ledger_entries=(self.ledger.get(family_id, [])
                if self.cfg.v3.diagnosis.expectation_ledger else []),
```

`self.ledger` is keyed **per family**. So a rewriter call sees a ledger only when *the same family* gets
a **second** rewrite round after its first was reconciled. That keying is correct — a family must not be
shown another family's predictions, which are about different code.

## No family has ever had a second round

Counted across every run on box 1, including both live arms
(`scripts/ledger_reach.py`, which groups `REWRITE_PRODUCED` by `(family_id, call timestamp)` because
each round produces two candidates and counting events would double it):

| run | families with a round | rounds each |
|---|---|---|
| `run-l3-21-20260908-232211` | 3 | 1, 1, 1 |
| `run-l3-48-20260907-202457` | 3 | 1, 1, 1 |
| `run-l3-48-20260909-115701` | 3 | 1, 1, 1 |
| `run-l3-43-20260911-053020` (live control) | 2 | 1, 1 |

**Nine families in finished runs plus both live arms: every one received exactly one round.** So the
`ledger_entries` argument has been `[]` on every rewriter call the project has ever made, in both arms,
regardless of the switch.

Verified directly on disk rather than inferred: every `rewriter-*` sandbox on both arms contains
`failed_hypotheses.json` (2 bytes, `[]`) and **no `prediction_ledger.md`** — including on box 2, which
has `expectation_ledger: true`.

## Why, arithmetically

`active_families()` rule 1, quoted from its own docstring: *"Every family that has never had a rewrite
round goes first. A branch may not be dropped before it has been given one chance to show its
headroom."* That rule exists for a measured reason — the recorded case where ranking families by latency
would have deleted the run's eventual winner.

The three finished runs are identical on the two numbers that matter:

```
run-l3-21-20260908-232211   families=4  seeds=4  rewrite_rounds=3
run-l3-48-20260907-202457   families=4  seeds=4  rewrite_rounds=3
run-l3-48-20260909-115701   families=4  seeds=4  rewrite_rounds=3
```

**4 seed families, 3 rounds.** The never-rewritten queue holds 4 entries and only 3 rounds fit in the
budget, so it never empties and no family is ever revisited. This is not a near miss that a slightly
longer run would fix by luck: it is off by one family, every time, on three independent runs.

Consistent with `wall-clock-is-always-the-binding-budget` (5 of 5 finished runs ended on the wall clock
having used only 1–2 of 5 rewrite rounds per family) — the budget is exhausted long before the rotation
comes around.

## What this means for the paper

**S2d(a) and S2d(b) are unaffected and are the parts with production evidence.** Declarations are made
before measurement and journalled per candidate (16 on each arm), and the reconciliation arithmetic runs
and is re-derivable. Those are real results.

**S2d(c) — that the ledger improves the next rewrite — has zero production evidence, and cannot get any
from a run of this shape.** Report it as untested rather than as tested-and-null: a treatment never
administered is not a treatment that did not work.

This also **retires the pooled-ledger defect's remaining cost.** That defect's only outlet was the next
round's prompt (`docs/result-s2d-pooled-ledger-defect.md`), and this measurement says the next round for
that family never comes. So the pooled entries cost the experiment **nothing at all** beyond a
mislabelled log entry that is re-derivable offline — which is a stronger statement than the "possible
but not certain" that document currently carries.

## What would be needed to test S2d(c)

Any of these changes the experiment, so none is applied mid-run:

1. **Fewer seed families** (`max_seed_candidates` 4 → 2). Then the queue empties after 2 rounds and
   round 3 revisits. Cheapest, and it trades structural breadth for depth — the opposite of what
   `active_families()` rule 1 was built to protect, so it needs its own justification.
2. **A longer budget.** At 3 rounds per 12 h, revisiting the first family needs 5 rounds ≈ 20 h. Direct
   but expensive, and the 4-vs-3 gap means it must clear the queue, not merely extend it.
3. **Reserve a round for a reconciled family** — make rule 1 yield once a family has an
   `EXPECTATIONS_RECONCILED` entry. This is the change that makes S2d(c) testable *by design* rather
   than by budget luck, and it is the one to specify properly after the runs finish, because it alters
   the search order the arms share.

## Guard

`scripts/ledger_reach.py` reproduces the table from any set of run directories and prints the single
line that matters: whether any family anywhere received a second round after a reconciliation. It
currently prints `NO -- S2d(c) has never been administered`.

**The conclusion is also encoded in the wrap-up report** (`check_wrapup.check_ledger_reach`, block
`[3c]`), because leaving it to my reading at report time was the actual risk: block `[3b]` prints
`PASS -- 1 entries over 1 rounds, 16 declarations reconciled` on exactly this state, and that PASS is
about journalling. Live-verified on both arms — each prints `**S2d(c) NEVER ADMINISTERED** -- 2
family(ies), max 1 round(s) each`, and the paired report adds the arm-level consequence: *the arms are
identical with respect to S2d(c), so any latency difference between them is not evidence about it.*

The reach test is **ordered, not a pair of counts**. `k >= 2 and reconciled >= 1` would call a family
reached when its only reconciliation landed at round 2's *close* — after the last prompt that could
have carried it — so the flag is read at each rewriter call's own position instead. And a round is one
rewriter **call**: 9/9 measured rounds produced two candidates, so counting `REWRITE_PRODUCED` events
would double every round and report a first round as a revisit, which is a fabricated positive rather
than a missed detection.

`scripts/revert_check_ledger_reach.py`, four variants, each probed on a fixture that reaches its own
branch (`one_round` / `reached` / `recon_last` — a single fixed probe cannot show all four, the
recorded SHAM failure mode). All CAUGHT: `round_counted_per_event`, `ordering_dropped`,
`reach_never_fires` (which makes the check unfalsifiable — it prints the conclusion I already believe),
`unreconciled_revisit_counts_as_reach`.
