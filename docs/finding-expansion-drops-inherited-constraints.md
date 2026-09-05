# Finding: a space expansion silently un-does the constraints it inherits

Improvement K's whole premise, in its own comment at `orchestrator.py:894`:

> "An expansion only ADDS choices, so the pre-expansion optimum is still a legal config"

The domain half of that is true and now measured. The **constraint** half was never
enforced, and it does not hold: across **all 30 expansions on record, the replacement
space admitted configurations its predecessor had excluded in 21 of them** — a mean
**12.4% of the sub-grid the two spaces share**.

Found live at 14:58 on `cand-e3a5da01`, the candidate that had just produced the
project's best result (9.73 ms). Its expansion kept 5 constraints and dropped 8,
including every shared-memory and thread bound.

## Why the agent does it, and why it is not the agent's fault

An expansion re-declares the whole space (`space.params` must list every knob, or the
response is rejected as `key_mismatch`). The prompt spells that requirement out for the
**domains** — "repeat the unexpanded ones verbatim", with the full inventory printed —
and says nothing equivalent for the **constraints**, because until now the directive did
not contain them.

And it could not derive them. Constraints live in the `ParameterSpace` object, not in
`candidate/source.py`. The sandbox holds `source.py`, `candidate_contract.md`,
`device.md` — the space's constraint list appears nowhere. The agent was being asked to
reproduce, from memory of a file it never saw, a list that determines whether 40 GPU
trials get spent on launchable configurations.

Every prior investigation of this pathway looked at the domains and stopped:
`measurement-per-knob-expansion-attribution.md`, `audit_expansion_outcomes.py`, and the
`no_new_choices` guard added at `orchestrator.py:866` all reason about `choices` only.

## What the relaxation costs, measured

A text diff of the constraint lists overstates this badly — one expansion re-expressed
three `<= MAX_THREADS_PER_BLOCK` bounds as a single `and`-chain and lost nothing, and by
text that reads as "3 dropped, 1 added". So the measurement samples the shared sub-grid
and compares **verdicts**:

```
                                        trials  complete   failed
only reachable because a constraint         58    51.7%    48.3%   (runtime_error 25)
  was dropped
legal under both old and new               662    74.0%    26.0%
```

Nearly **twice the failure rate**, and the failures are `runtime_error` — launches that
died on resources the dropped bound existed to respect.

Per candidate, comparing only within a candidate (so tasks are never mixed):

| | n |
|---|---|
| candidates where the newly-admitted region held a **better** latency | **0 of 21** |
| tie (within 2%) | 3 |
| strictly worse, or every trial in that region failed | **18** |

So the relaxation is not a hidden benefit being clawed back. It bought nothing on any
candidate in any run, and it spent trials to do it.

## What is NOT wrong

Two invariants I expected to be broken are intact, and it matters that they are checked
rather than assumed:

- **The incumbent carry-forward is safe: 30 of 30.** The pre-expansion best config still
  passes the guard against the new space every time, so the anchor added at
  `orchestrator.py:904` (after `cand-0c3b5820` went 20.0 → 22.6) always works.
- **The domain is a superset: 29 of 30.** One expansion dropped `EXPAND_BLOCK_N=32`,
  which the `NEVER SHRINK A KNOB` rule already forbids in prose.

## The fix, in two parts

**1. Tell the agent what it is being asked to preserve** (`modules.py`). The expansion
directive now carries the prior constraints with their rationales, stating that this
listing is the only place they can be read, that each still-applicable one must be
repeated verbatim, and that a constraint contradicted by a rewritten body should be
*replaced with a corrected form* rather than omitted.

**2. Restore what is dropped anyway** (`orchestrator.py::_restore_dropped_constraints`).
A prompt cannot guarantee compliance, so the driver re-adds any prior constraint the
response omitted — **gated on an empirical test**: re-add it only if every newly added
choice still has at least one feasible configuration under the restored set.

That gate is the part that makes this safe rather than merely strict, and it earns its
keep on real data. `cand-88e76051`'s expansion is the counter-case: its body was
rewritten so an N tile of 8 became legal, `BLOCK_N=8` was the value being added, and the
inherited `BLOCK_M % 16 == 0 and BLOCK_N % 16 == 0` would have vetoed exactly that. On
that pair the fix restores the two resource bounds and correctly declines the stale
legality rule:

```
RESTORED:  + NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK
           + BLOCK_M * BLOCK_N <= 4096
DECLINED:  - BLOCK_M % 16 == 0 and BLOCK_N % 16 == 0    (would forbid BLOCK_N=8)
BLOCK_N=8 reachable after restore: True
```

A stale constraint vetoing the expansion it was requested for would be a worse bug than
the one being fixed, which is why the gate is empirical rather than a heuristic about
which constraints "look like" resource bounds.

## Replayed over every recorded expansion

```
SEMANTIC   expansions that admit configs the old space excluded: 21/30  (mean 12.4%)
AFTER FIX  same, with dropped constraints restored:               0/30  (mean  0.0%)
           constraints restored: 119
           newly added choices made unreachable:  0 of 80
```

Both halves matter: the leak closes completely, and **not one of the 80 newly added
choices** across all 30 expansions becomes unreachable. The fix is a pure tightening
back to the inherited semantics, with no loss of the search range the expansion was
requested for.

Reproduce with `python scripts/audit_expansion_relaxation.py`. Unit tests pin all three
behaviours (restore, decline-when-stale, no-duplicate) plus the prompt content, using the
two real shapes from disk as the cases.

## Scope and honest limits

- **Driver-side, so it does not affect the running L3:43 experiment.** `cand-e3a5da01`'s
  expansion already happened under the old behaviour. It applies from the next run.
- **The 9.73 ms result is unaffected either way.** The expanded space re-tuned to the
  same 9.73 (the carried anchor rediscovered it) and `improved_family: false`, so nothing
  the relaxation admitted entered the result. What it cost was 40 trials, 20 of which
  failed — 15 `runtime_error`.
- **`_choice_is_reachable` is a random search, not a proof.** 2000 samples per newly
  added choice; a false "unreachable" declines a restore, which is exactly the
  pre-existing behaviour, so the failure mode is inert.
- **The 0-of-21 result is not a claim that constraints can never be too tight.** It says
  that on this record, the configurations the dropped constraints had been excluding were
  not where the wins were. A genuinely over-tight inherited constraint is still possible;
  the reachability gate is what handles that case, and it fired once in 30.
