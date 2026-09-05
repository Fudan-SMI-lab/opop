# Result: seeding `best_history` makes `converged` reachable on 7 of 15 families, and cuts no live search

`docs/plan-next-round-four-fixes.md` predicted that seeding the seed-phase best as
`best_history[0]` would make `stop_kind="converged"` fire for the first time. Replayed
through the **real** `ConvergencePolicy` against every family on disk that ran enough
rounds for the test to differ:

```
families examined:                        15
newly reach stop_kind='converged':         7
freeze one round EARLIER (saved GPU time): 7
families still improving when frozen:      0
```

| run | family | seed | rounds | unseeded | seeded |
|---|---|---|---|---|---|
| l3-21 0902 | fam-3dacc96b | 25.2 | [25.2, 25.2, 25.2] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0903 | fam-97b5e645 | 24.6 | [24.6, 24.6, 24.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0903 | fam-eb982220 | 24.6 | [24.6, 24.6, 24.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-21 0904 | fam-f14a294f | 21.7 | [21.3, 20.5, 20.5] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-21 0904 | fam-b278c226 | 21.9 | [21.5, 20.7, 20.7] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0903 | fam-59e65a76 | 31.6 | [31.6, 31.6, 31.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0903 | fam-d4be7a3a | 29.3 | [29.3, 29.3, 29.3] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0903 | fam-36474be2 | 21.0 | [19.4, 19.4, 19.4] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0903 | fam-19e13c64 | 22.9 | [19.5, 19.5, 19.5] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0904 | fam-c9461c56 | 19.6 | [19.6, 19.6, 19.6] | round 4: budget_exhausted | **round 3: converged** |
| l3-43 0904 | fam-ff3ef34b | 20.0 | [19.5, 17.9, 17.9] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0905 | fam-92e7c576 | 22.5 | [19.6, 19.6, 19.6] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-43 0905 | fam-4aea322a | 14.2 | [11.0, 11.0, 9.7] | round 4: budget_exhausted | round 4: budget_exhausted |
| l3-48 0905 | fam-99aee6de | 2.1 | [2.1, 2.1, 2.1] | round 4: budget_exhausted | **round 3: converged** |
| l3-48 0905 | fam-74c41d8d | 3.5 | [2.5, 2.1, 2.1] | round 4: budget_exhausted | round 4: budget_exhausted |

## The safety property, which is the part that matters

The risk with reaching `converged` sooner is deleting structural search — the failure this
project exists to avoid. It does not happen here, and the pattern in the table says why:

**Every one of the 7 newly-converged families is flat across all three rounds** — 25.2/25.2/25.2,
24.6/24.6/24.6, 31.6/31.6/31.6, 19.6/19.6/19.6, 2.1/2.1/2.1. Their seed equals their
final best; the rewrites achieved nothing measurable.

**Every family that improved keeps its full budget**, including the two that mattered most:
`fam-4aea322a` (14.2 → 11.0 → 11.0 → 9.73, the project's best L3:43 result) and
`fam-ff3ef34b` (20.0 → 19.5 → 17.9, which produced L3:43's earlier winner). Both still
freeze at round 4 on `budget_exhausted`, unchanged.

`scripts/audit_history_seeding.py` checks this explicitly and exits 1 if any family still
gaining ≥ `min_improvement_pct` is frozen earlier. It exits 0.

## A methodological correction worth recording

My first replay of this reported **0 changes** and I nearly filed the fix as inert. The
error: it evaluated each family once at `used == len(history)`, i.e. *after* its last
round, when the verdict that decides anything is taken **before** each round
(`_rewrite_round` calls `family_verdict` at the top of the loop). Walking the rounds
properly shows 7 of 15.

The wrong version is a plausible-looking snippet that produces a confident null result,
which is why the walk now lives in a script with the ordering spelled out in its
docstring rather than being re-derived each time.

## What this does and does not buy

- **A real `converged` label**, so the report can distinguish "this branch stopped paying"
  from "this branch ran out of rounds". Those were indistinguishable before
  (`finding-converged-stop-kind-is-unreachable.md`).
- **One rewrite round saved on 7 of 15 families.** At the observed 0.5–1.1h per family
  round on L3:21 that is real time, redirected to families that are still moving.
- **It does not change any reported latency.** Every family in the table reaches the same
  best either way; the seeded runs simply stop describing a flat branch as unfinished.

## The second half, now measured: the slope is not merely stale, it is *absent*

I wrote above that "the second half of the fix — the slope no longer being one round stale — is
not measured here", and described it as an ordering effect only visible in a live run. That
undersold it, and reading `_improvement_pct` says why:

```python
hist = f.best_history
if len(hist) < 2:
    return 0.0
```

Unseeded, at the decision that selects families for **round 2**, every family has run exactly
one round and therefore has a **one-entry** history. All of them return slope `0.0`, tie, and
fall through to `rank()`'s third key — **absolute latency**. That is precisely the
early-pruning-by-latency the docstring spends two paragraphs rejecting. So the slope ordering
was not one round stale at that decision; it was unavailable, and the documented fallback was
the ranking it exists to replace.

Replayed over all 11 finished runs (`scripts/audit_slope_ordering.py`), the round-2 selection
**set** differs in 1. That single case is worth reading rather than dismissing:

```
run-l3-43-20260905-091705, max_families_active=2
  fam-92e7c576  seed 22.5 -> round1 19.6   slope unseeded 0.00%  seeded 12.89%
  fam-4aea322a  seed 14.2 -> round1 11.0   slope unseeded 0.00%  seeded 22.54%
  fam-ea7bc8bb  seed 28.6 -> round1 21.3   slope unseeded 0.00%  seeded 25.52%
  fam-7f682a54  seed 23.5 -> round1 19.9   slope unseeded 0.00%  seeded 15.32%
chosen unseeded (latency tie-break) : fam-4aea322a, fam-92e7c576
chosen seeded   (real slope)        : fam-ea7bc8bb, fam-4aea322a
```

All four slopes collapse to 0.00% unseeded — the mechanism, shown rather than argued. And the
family the latency tie-break selected, `fam-92e7c576`, is the one that went
**19.6 → 19.6 → 19.6**: three rounds, zero gain. What it consumed:

```
round timeline (hours from run start)
  2.44h  fam-92e7c576  round 1  19.6
  5.43h  fam-92e7c576  round 3  19.6
  6.24h  fam-92e7c576  round 4  19.6      <- run ended at 6.235h
trials spent on that family after its round 1: 160  (97 completed)
```

Meanwhile `fam-ea7bc8bb` (the highest seeded slope, 25.52%) and `fam-7f682a54` each got **one**
round and were frozen on budget with the run's 12 rewrite rounds only 8 spent. So the freed
budget had somewhere to go.

**What this does and does not establish.** The audit exits 0 and prints "no case where the
unseeded tie-break drops a family that later improved" — that check asks whether the *dropped*
family later improved, and `fam-92e7c576` is not dropped by the seeded order in the harmful
direction. What is demonstrated is the weaker, still useful claim: the tie-break selected the
branch that turned out flat over one with 2× its measured slope, and the flat branch then spent
160 trials confirming it was flat. Whether `fam-ea7bc8bb` would have gained anything with those
rounds is **not knowable from this record** — it was never given them. I am not going to claim
a latency win that the disk cannot support.

n=1 of 11 is also a small effect. It is small for a structural reason worth stating: the set
only changes when the latency order and the slope order disagree *across the K boundary*, which
needs at least three families and a genuine slope spread. Most runs have fewer families active
than that.

## Live confirmation on `run-l3-21-20260905-195615`

`FAMILY_SEEDED` fired for all four families, and the first family verdict on disk reads:

```
CONVERGENCE_DECIDED  family  continue  stop_kind=None  best_history=[19.4]  rewrite_rounds_used=0
```

Before the fix that history was `[]` — the state in which `converged` is arithmetically
unreachable. So the seed datum is present at a real decision point, not just in replay.

Then `fam-a4a8353c` completed round 1 at **11.0 ms** from a 19.4 ms seed
(`FAMILY_ROUND_RECORDED  round=1  best_ms=11.0`). Its slope entering round 2:

```
with the seed    : best_history=[19.4, 11.0]  ->  43.3%
without the seed : best_history=[11.0]        ->   0.0%
```

The run's strongest branch — the one that produced the 11.4 ms verified result — would have
scored **zero measurable headroom** at the moment round 2 was allocated. That is the mechanism
stated as sharply as the data allows.

**What it did not change, this time.** Both orderings pick the same two families right now, and
for a reason the fix does not touch: `rank()`'s first key is "never had a rewrite round", and
three families are still unproven, so they go first regardless of slope. The seeding will only
bind once all four have a round each. Worth stating plainly rather than claiming a live win — the
43.3% figure shows the slope is *available*; it has not yet been *used*.

Reproduce with `python scripts/audit_slope_ordering.py`.

Reproduce with `python scripts/audit_history_seeding.py`.
