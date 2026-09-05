# Finding: `stop_kind = "converged"` is structurally unreachable

Measured, not inferred: **13 of 13 family freezes across every L3 run are
`budget_exhausted`. `converged` has never fired once**, including eight families whose history
is completely flat.

```
run-l3-21-20260902-113144  fam-3dacc96b  budget_exhausted  [25.2, 25.2, 25.2]
run-l3-21-20260903-071357  fam-97b5e645  budget_exhausted  [24.6, 24.6, 24.6]
run-l3-21-20260903-071357  fam-eb982220  budget_exhausted  [24.6, 24.6, 24.6]
run-l3-21-20260904-013056  fam-f14a294f  budget_exhausted  [21.3, 20.5, 20.5]
run-l3-21-20260904-013056  fam-b278c226  budget_exhausted  [21.5, 20.7, 20.7]
run-l3-43-20260903-020233  fam-d4be7a3a  budget_exhausted  [29.3, 29.3, 29.3]
run-l3-43-20260903-020233  fam-59e65a76  budget_exhausted  [31.6, 31.6, 31.6]
run-l3-43-20260903-145357  fam-36474be2  budget_exhausted  [19.4, 19.4, 19.4]
run-l3-43-20260903-145357  fam-19e13c64  budget_exhausted  [19.5, 19.5, 19.5]
run-l3-43-20260904-093730  fam-ff3ef34b  budget_exhausted  [19.5, 17.9, 17.9]
run-l3-43-20260904-093730  fam-c9461c56  budget_exhausted  [19.6, 19.6, 19.6]
run-l3-48-20260905-010737  fam-99aee6de  budget_exhausted  [2.09, 2.09, 2.09]
run-l3-48-20260905-010737  fam-74c41d8d  budget_exhausted  [2.46, 2.09, 2.09]
```

`[25.2, 25.2, 25.2]` and `[2.09, 2.09, 2.09]` are families that did not move for three
consecutive rounds and were still recorded as having run out of budget rather than converged.
(Verified with `scripts/audit_convergence_stop_kinds.py`; the count was 11 when first written and
grew with the L3:48 run — the *ratio* has never changed, because the cause is arithmetic.)

## The mechanism

Two independent off-by-ones compound.

**1. `best_history` excludes the seed.** It starts empty and gains one entry per *completed
rewrite round* (`families.py:260`). The seed's tuned latency — which is the baseline every
improvement is measured against — is never in it. So `[2.09, 2.09]` does not mean "two
measurements"; it means seed → round 1 → round 2 with **two** no-improvement transitions
already observed.

**2. The budget check runs first, and both conditions arrive together.** In
`_rewrite_round`:

```
1. verdict = family_verdict(family)     <-- reads best_history as it currently is
2. if freeze: continue
3. _do_rewrite(...)
4. family.rewrite_rounds_used += 1
5. record_round(...)  -> best_history.append(...)
```

At the check, `len(best_history) == rewrite_rounds_used`. `family_verdict`
(`convergence.py:21-36`) tests, in order:

- freeze on budget when `rewrite_rounds_used >= rewrite_rounds_per_family` → **3**
- freeze on converged when `len(history) >= no_improve_rounds + 1` → **3**

Both become true at the same moment and the budget test is first. So:

> `converged` is unreachable whenever `no_improve_rounds + 1 >= rewrite_rounds_per_family`.

At the L3 config — `no_improve_rounds: 2`, `rewrite_rounds_per_family: 3` — that is `3 >= 3`.
Unreachable, and the smoke configs are tighter still.

## Why it matters

Three consequences, in increasing order of importance:

**A family always spends its last rewrite round.** A demonstrably stalled family cannot be
frozen early, so the third round is always paid for even when the first two produced nothing.
On L3:48 `fam-99aee6de` was 2.09 → 2.09 → 2.09 and still received a third round. At roughly
30-40 min per round that is real budget, taken from families that were still moving.

**And the third round has never paid off — 0 for 13.** Counting improving steps (>0.5%) across
every recorded family history:

| step | improved | flat |
|---|---|---|
| round 1 → round 2 | **4 of 13** (3.8%, 3.7%, 8.2%, 15.0%) | 9 |
| round 2 → round 3 | **0 of 13** | 13 |

Not one family in any run has improved on its third round. Combined with the unreachable
`converged` verdict, that is the sharpest statement of the cost: the harness is structurally
obliged to spend a round that has a 0-for-13 record, on every family, in every run — roughly
1.5 h per run at ~22 min per round across four families.

This strengthens the case for the fix rather than changing it: with `best_history` seeded, a
family whose first two rounds gained less than `min_improvement_pct` would freeze at the right
time and the dead third round would be skipped. Note the caveat below about a 2.1% re-tune swing
against a 2.0% threshold still applies — but the 0-for-13 record means the risk of freezing one
round *too early* is much smaller than the certainty of wasting one round *too late*.

**The reported `stop_kind` is wrong.** Every family in every report reads
`budget_exhausted`, which tells a reader "we ran out of time here, there may be more
headroom" when the truth for six of these eleven is "three rounds produced no improvement".
That is the opposite conclusion.

**It is a paper-facing claim.** `ConvergenceDecision.stop_kind` exists precisely to
distinguish *converged* from *out of budget* — the convergence half of the two-loop argument.
The field is implemented, tested, and has never once taken the value the argument needs.
Reporting convergence behaviour from these runs would mean reporting a code path that never
executed.

## A third, related off-by-one

`FamilyManager._improvement_pct` (`families.py:236-249`) — the slope used by
`active_families()` to rank who gets the next round — also needs 2 entries, so it returns
**0.0 for a family that has completed only one rewrite round**. A family that improved
sharply in its first round is ranked as though it had stalled, and loses to any family
already holding two entries. Same root cause: the seed is not in the history, so the first
round produces no measurable slope.

Note this does *not* invalidate `result-second-ranked-family-catches-the-leader.md`: there,
both families entering round 2 had exactly one entry, so both scored 0.0 slope and the
ordering came from the unproven-first rule plus latency. The documented round-2 counterfactual
(stalled leader 0% vs moving challenger +13.4%) is computed from the *recorded histories*,
which is the right comparison for the claim — but the ranking code at that moment saw 0.0 for
both. The claim stands; how it was reached is partly this bug.

## Watched live, on the best result the project has produced

`run-l3-43-20260905-091705`, `fam-4aea322a`, 11:31:09. This is the first time both off-by-ones
have been observed *as they happened* rather than reconstructed from a finished run, and the
family in question is the one that produced the run's 11.0 ms.

```
FAMILY_ROUND_RECORDED  {"family_id": "fam-4aea322a", "best_ms": 11.0, "round": 1}
CONVERGENCE_DECIDED    {"scope": "family", "verdict": "continue",
                        "evidence": {"best_history": [], "rewrite_rounds_used": 0}}
```

The family went **14.2 → 11.0 in its first round, a 22.5% improvement** — and the convergence
evidence at that moment reads `best_history: []`. Two consequences visible in one event pair:

1. **The 22.5% gain is invisible to the policy.** `best_history` is empty at the check, so
   `_improvement_pct` returns 0.0 and this family — the fastest-improving one on record for this
   task — is ranked for the next round as though it had stalled completely.
2. **The seed's 14.2 is nowhere.** With the seed included, the history would read `[14.2, 11.0]`
   and the improvement would be measurable after round 1, which is what `active_families()` was
   written to consume.

Note the `CONVERGENCE_DECIDED` for `fam-92e7c576` at the same timestamp *also* reads
`best_history: [], rewrite_rounds_used: 0` — correct there, since that family has genuinely had
no round yet. The two are indistinguishable in the event log: a family that improved 22.5% and a
family that has done nothing both report `[]`.

### The unreachability, proved arithmetically rather than argued

At the L3 config (`rewrite_rounds_per_family: 3`, `no_improve_rounds: 2`), enumerating the state
at each check:

| at start of round | `rounds_used` | entries in history | budget freeze? | can even test converged? |
|---|---|---|---|---|
| 2 | 1 | 1 | no | no (needs 3) |
| 3 | 2 | 2 | no | no (needs 3) |
| 4 | 3 | 3 | **yes** | yes — but never reached |

The improvement test needs 3 entries, which requires `rounds_used >= 3`, which triggers
`budget_exhausted` first *in the same call*. There is no configuration of this run in which
`converged` can fire. That is not a probabilistic claim about these runs — it is arithmetic, and
it is why the count below is 13 of 13 rather than merely lopsided.

### The invariant the proof rests on, now machine-checked

The argument above assumes `len(best_history) == rewrite_rounds_used` at every check. That is
worth verifying rather than asserting, since it is the single load-bearing step. Checking **every
family `CONVERGENCE_DECIDED` in every run**:

```
family decisions checked: 96 | len(best_history) != rewrite_rounds_used in 1
```

The one exception is `run-l1-19-20260902-011132` `fam-daf4267d` — `best_history: []` with
`rewrite_rounds_used: 1` — and it is not a counter-example: that smoke run has
`rewrite_rounds_per_family: 1`, the family is empty (`best is None`), and it took the
empty-family branch that increments the counter without recording a round
(`finding-run-stops-with-budget-unused.md`). So the invariant holds in all 95 cases where a
round was actually recorded, which is exactly the set the proof concerns.

`record_round` (`families.py:261`) is a bare `append` called once per round, so this is
structural rather than incidental.

### And the corollary, observed live at 14:02:50

`run-l3-43-20260905-091705`, `fam-4aea322a`, second round recorded:

```
FAMILY_ROUND_RECORDED  {"family_id": "fam-4aea322a", "best_ms": 11.0, "round": 3}
```

Flat — 11.0 after round 1, 11.0 after round 2 (both round-2 children failed to beat it:
`cand-aa016dfe` 11.8, `cand-45c3fd7d` 20.3, see
`result-analyst-hypothesis-refuted-by-control.md`). **This was pre-registered before the event
fired**, and it is the first family in this run to reach a two-entry history.

So `fam-4aea322a` now sits at `[11.0, 11.0]` with `rewrite_rounds_used: 2` — one genuine
no-improvement transition observed, `_improvement_pct` computing a *true* 0.0% for the first
time in this run rather than the usual one-entry artifact. Per the table above it needs a third
entry to test `converged`, and the third entry arrives with `rounds_used: 3`, so this family will
freeze as `budget_exhausted` after spending a round with a 0-for-13 record. That will make it
**14 of 14**.

One correction to what I said when pre-registering this: I called `[11.0, 11.0]` "the first
`best_history` in the project long enough for `_improvement_pct` to compute anything". That is
wrong — thirteen length-3 histories exist (the table at the top of this file). What is true is
narrower: it is the first in *this run*, and `_improvement_pct` has been computing real slopes
since 09-02. The unreachability was never about histories being too short in general; it is
specifically that a history reaches length 3 only in the same call that exhausts the budget.

## The fix

Seed `best_history` with the family's post-tuning incumbent at the moment it enters Loop C,
so the list means "incumbent after each round, including round 0". Then:

- `[2.09(seed), 2.09, 2.09]` has 3 entries after 2 rounds → `converged` fires at the right
  time and the third round is saved;
- `_improvement_pct` returns a real slope after the *first* round, which is what
  `active_families()` was written to use;
- `budget_exhausted` goes back to meaning what it says.

This changes when families freeze, so it must not go into a run already in flight — it is
driver-side and applies to the next run. It also interacts with
`measurement-retune-repeatability.md`: a 2.1% re-tune swing sits right on
`min_improvement_pct = 2.0`, so a noisy re-tune could now trigger `converged` one round early.
That risk is the mirror of today's behaviour (always one round late) and is the reason the
retune doc declined to raise the threshold; worth revisiting together once per-sample timing
lands.

**Not applied yet** — unlike the fp16 witness fix, this one changes how much structural
search each family gets, which is a budget-allocation decision rather than a bug in a
diagnostic. Flagging it with the evidence; it belongs in the same batch as the gate decision.
