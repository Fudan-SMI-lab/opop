# Measurement: per-knob attribution splits K's expansions cleanly — tiles up, warps never

`scripts/audit_expansion_outcomes.py` now attributes each expansion **per widened knob**, not
just per expansion. The motivating case was `cand-45c3fd7d`, whose single expansion widened seven
knobs and was scored as one 7.7% success while containing at least one clearly counterproductive
widening (`finding-parameterizer-lacks-triton-pitfalls-doc.md`).

## The headline hides the composition

`cand-45c3fd7d`, `sp-68a83210` → `sp-f5e27f18`, 22.0 → 20.3 (**+7.7%, attributable** — the
winning trial uses the newly-added `OUT_BLOCK_M=256`). Same expansion, per knob:

| knob | added | median with | median without | verdict |
|---|---|---|---|---|
| `OUT_BLOCK_M` | 256 | — | — | winner used it |
| `QKV_NUM_WARPS` | 2 | 37.7 (n=2) | 24.4 (n=19) | **−54.5%** |
| `QKV_BLOCK_K` | 128 | 33.1 (n=2) | 24.5 (n=19) | **−34.9%** |
| `SCORE_NUM_WARPS` | 16 | | | hurt |
| `PV_BLOCK_K` | **8** | 24.5 (n=3) | 20.3 | hurt, and cost a rejected attempt |

One expansion, one knob earning the gain, four knobs adding slower regions to a 40-trial budget.
Nothing in the current metric distinguishes this from an expansion where all seven helped.

## A methodology error I made first, and had to fix

My first implementation compared the **minimum** of trials using an added value against the
minimum of those not using it, and reported **12 helped / 38 hurt (73% hurt)**. That number is an
artifact. The "did not use it" pool is every other trial in the re-tune, so it is systematically
larger:

```
rated knobs: 52
median n_with    =  4
median n_without = 19
n_with < n_without in 47 of 52 cases
```

`min` over a 19-sample pool is lower than `min` over a 4-sample pool for free, regardless of which
region is better. Switching to **medians** moves the same 52 knobs to **24 helped / 4 tied / 24
hurt** — from "K's widenings are usually harmful" to "coin flip". The script now compares medians
and says why in a comment, so the mistake is not re-introduced.

This is the same failure mode as the two errors recorded in
`finding-k-retune-cannot-disconfirm-its-incumbent.md`: reading a rate off a statistic whose
denominator I had not examined.

## The size-independent test agrees, and is the one that matters

Sample size cannot bias "did the expansion's own winning trial use one of this knob's added
values?" — that is the single decision the search acted on:

```
YES 12 (23%)   NO 40 (77%)
```

So **77% of widened knobs are not the reason their expansion improved**, even in the expansions
that did improve. That is consistent with the per-expansion attribution already on record
(48% of expansions attributable to an added value) and sharpens it: within an attributable
expansion, usually exactly one knob earns it.

## The real signal: direction and knob kind

Splitting the 52 rated knobs by whether the added values sit above or below the old domain:

| direction | helped | hurt/tied | rate |
|---|---|---|---|
| **UP** (larger values) | 20 | 14 | **59%** |
| **DOWN** (smaller values) | 4 | 14 | **22%** |

And by what the knob controls:

| direction | kind | helped | hurt | rate |
|---|---|---|---|---|
| UP | **tile size** | **16** | 5 | **76%** |
| UP | warp count | 2 | 7 | 22% |
| UP | stage count | 2 | 2 | 50% |
| DOWN | tile size | 2 | 4 | 33% |
| DOWN | **warp count** | **0** | **7** | **0%** |
| DOWN | stage count | 2 | 3 | 40% |

Two clean readings:

1. **Widening a tile size upward is K's productive case: 16 of 21 (76%).** This is the mechanism
   the paper describes working as intended, and it matches the independent finding that row tiles
   are monotone with an optimum at the largest feasible value
   (`result-row-tile-is-monotone-and-k-supplies-it.md`).
2. **Widening a warp count helps 2 of 16 overall and 0 of 7 downward.** Every `NUM_WARPS += [1]`
   and `+= [16, 32]` in the record made things worse or no better. The failures are large, not
   marginal — `OUT_NUM_WARPS += [1]` gives a median of 136.4 against 39.7, and
   `STATS_NUM_WARPS += [16]` 89.1 against 29.2.

The warp result has a plausible mechanism: 1 warp serialises a tile that was written expecting
several, and 16–32 warps (512–1024 threads) exceeds what these kernels' register budgets can
sustain per block, so occupancy collapses. Neither end is a region a tile-shaped kernel benefits
from, whereas a *tile* knob's extremes trade a real quantity (reuse vs resources).

## What this supports, and what it does not

It is independent evidence for **pending item 8** (per-knob floors in K's expansion logic), which
I had proposed from the `NUM_STAGES -> 0` cases alone. Those five cases are visible here too
(`NUM_STAGES += [0]`, −38.7%), but the warp-count result is a larger and cleaner effect that the
same mechanism would address — a floor of 2 on any `*_WARPS` knob would have prevented all seven
0-for-7 downward warp expansions.

It does **not** establish that K should stop expanding warp counts. Two reasons to hold back:

- The 52 rated knobs come from 28 expansions across 6 runs, and 13 more are unrated because one
  side was never freshly measured. The warp subset is n=16.
- "Helped" here is a median comparison within one re-tune's 40 trials, which is a weaker
  instrument than the per-expansion attribution. A knob whose added value is bad in most
  combinations can still be the one that unlocks the single best point — that is exactly what
  `QKV_NUM_WARPS += [16]` did for `cand-cb7be6b4` (+21.3% on medians *and* used by the winner),
  in the same knob family that fails 7 of 9 elsewhere.

So the honest summary is: **K's upward tile expansions are earning their place, its warp
expansions are not, and per-knob floors are the smallest change that would act on that.** Still
pending a user decision; nothing about K's logic has been touched.

Reproduce with `python scripts/audit_expansion_outcomes.py` (the `PER-KNOB` section).
