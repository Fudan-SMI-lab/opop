# Finding: `stop_kind = "converged"` is structurally unreachable

Measured, not inferred: **11 of 11 family freezes across every L3 run are
`budget_exhausted`. `converged` has never fired once**, including six families whose history
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
```

`[25.2, 25.2, 25.2]` is a family that did not move for three consecutive rounds and was
still recorded as having run out of budget rather than converged.

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
