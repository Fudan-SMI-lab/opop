# Finding: a K re-tune cannot disconfirm its incumbent — the cache guarantees the same number

Found while trying to reproduce L3:43's 14.2 ms outlier. The measurement cache is working
exactly as designed, but it has a consequence for how improvement-K results must be read,
and it partially invalidates the way I characterized two earlier "flat" expansions.

**Scope correction (see "The base rate" below).** An earlier version of this doc said a flat
re-tune "is the default outcome". Across the full record that is **false**: 15 of 23
expansions improved the reported best, 6 were flat, 2 got worse. The surviving claim is
narrower and still load-bearing: *when* a re-tune comes back flat, that flatness is not
evidence about the widened region, and no re-tune can ever re-measure its own incumbent.

## The mechanism

`_tune` caches measurements by parameter vector (`measured_cache`, journalled as
`reused_measurement: True` since `ea9a3a2`). A K expansion widens *some* knobs' domains and
leaves the rest alone, so the incumbent θ_best is still a legal point in the new space — and
when TPE samples it again, it is a **cache hit**, not a measurement.

Therefore: if the pre-expansion best config is resampled in the re-tune, the post-expansion
best is **arithmetically guaranteed** to be ≤ the pre-expansion best, and equal to it unless
some genuinely new point wins. A "flat" re-tune is therefore uninformative about the widened
region — but it is not the *likely* outcome, and I was wrong to say so.

## The base rate — what actually happens across all 25 expansions

Every K expansion in the record that has a before/after `TUNING_DONE` pair:

| outcome | n | post-expansion best was a cache hit |
|---|---|---|
| **improved** | **17** | 0 |
| flat | 6 | **3** |
| worse | 2 | 0 |

Three things follow, and the first two contradict what I had written:

1. **Expansions usually help.** 17 of 25, ranging +0.4% to +20.0%. Improvement K is doing
   real work; the cache issue is a reading hazard, not a verdict on the mechanism.
2. **Flatness is uncommon (6 of 25) and only half of it is a cache artifact.** Three flat
   outcomes had a *freshly measured* best that merely tied — so even "flat" does not reliably
   imply "replayed".
3. **Two expansions made the reported best worse** (`cand-0c3b5820` 20.0 → 22.6,
   `cand-cb7a07c1` 22.3 → 22.8), neither from cache. That is only possible if the incumbent
   vector was *never resampled* in the 40-trial re-tune — which shows the cache-replay
   guarantee is conditional on TPE happening to revisit it, and with a widened space it often
   does not.

   **Checked: neither one damaged anything.** Both are `improved_family: False`, `crun.best_ms`
   keeps the better earlier value by the explicit guard at `orchestrator.py:711`, and
   `FamilyManager.update_best` (`families.py:254`) is monotonic. Their family's recorded
   history is `[19.5, 17.9, 17.9]` and the run's published best is a different candidate at
   17.9 tuned / 19.1 re-eval. So a worse post-expansion `TUNING_DONE` is a *reporting* artifact
   of that one event, not a regression in what the run keeps — the monotonic guards were
   already there and the code comment at `orchestrator.py:708` says why.

**Caution on reading that 68% as K's success rate.** The "improved" column counts whether the
post-expansion *best* fell, not whether K's *added values* caused it. A re-tune re-runs 40 trials
over the whole space, so it can find a better point inside the original domain and be scored as an
improvement. At least one case is now confirmed to be exactly that (`cand-88e76051`, +2.0% while
the added value landed 13% worse — see
`measurement-k-expands-downward-and-adds-zero-stages.md`), so **68% is an upper bound on K's
contribution**, and the honest per-case test remains the new-choice subset.

## Measured: which reported bests came from cache

Across every run, **3 reported bests came from a cached measurement** — all three are K
re-tunes, two on L3:21 09-05 and one on L3:43 09-05 (see the third instance below):

| candidate | pre-expansion space | post-expansion space | identical? |
|---|---|---|---|
| `cand-d31b0474` | `sp-df786ac5` = **17.1** | `sp-1636fa27` = **17.1** | **yes** |
| `cand-c0b3b7cd` | `sp-2f309139` = **25.0** | `sp-155aa1e4` = **25.0** | **yes** |

Both post-expansion bests are cache replays of the pre-expansion winner. The 17.1 → 17.1 and
25.0 → 25.0 "flatness" was not measured twice; it was measured once and reported twice.

## What this corrects

`measurement-k-expansion-vs-analyst.md` reads the `cand-d31b0474` case as an empirical
confirmation of the analyst's prediction:

> "40 trials, and the result was **17.10 → 17.10 ms, exactly flat**… Eight completed trials
> is enough to say the widened region is genuinely unproductive, not merely unlucky."

The second half stands — 8 completed trials that *did* use a new choice all landed at 18.20
ms or worse, and that is real evidence about the widened region. But the headline framing
("exactly flat") is weaker than I made it: the flat top-line number was structurally
inevitable once the incumbent was resampled, so it corroborates nothing on its own. The
informative statistic was always the new-choice subset, not the equality of the two bests.

Likewise `finding-optimization-behind-a-dead-mode-branch.md` notes of `cand-c0b3b7cd`:

> "Both budgets measured the same dead path, and both returned exactly 25.0 ms."

Still true and still damning about the dead branch, but "both returned exactly 25.0" is
partly a cache artifact rather than two independent confirmations.

## And it closes off one way of checking a suspicious measurement

The immediate reason I noticed: L3:43's `cand-cb7be6b4` reported 14.2 ms from a single trial
with a 2.1× gap to second place (`inprogress-l3-43-14ms-outlier.md`). A K expansion fired,
the re-tune sampled `ATTN_BLOCK_M=128` again, and it reported 14.2 ms — which looked like
independent reproduction.

It was not:

```
tr-d1701cb3  space=sp-7e04deb1  complete  14.2 ms std=0.41 min=13.0 max=14.5 n=20  (job files on disk)
tr-ebf1aa16  space=sp-db2e9791  complete  14.2 ms std=0.41 min=13.0 max=14.5 n=20  reused_measurement: True, NO job files
```

Identical to every digit, because it is the same measurement. The expansion widened
`QKV_BLOCK_K`, `QKV_NUM_WARPS`, `ATTN_BLOCK_N`, `OUT_BLOCK_M`, `OUT_NUM_WARPS` — and left
`ATTN_BLOCK_M` alone, so the winning vector was unchanged and cached.

**A re-tune can therefore never reproduce or disconfirm a suspicious best.** Only
`final_reeval_ms`, which runs in a fresh process against fresh inputs, does that.

## Why the cache is still right

Not a bug, and not something to remove. Re-measuring an identical parameter vector costs a
full GPU eval (~8–11 s here) to learn nothing new, and across a 40-trial re-tune that is
minutes of duplicated work. The journalling added in `ea9a3a2` is exactly what made this
diagnosable — without `reused_measurement` in the event, the two 14.2 ms trials are
indistinguishable from a genuine replication.

## What I would change — not applied

Reporting only, no semantics:

1. **Mark a cached best in the report.** When a `TUNING_DONE`'s best trial carries
   `reused_measurement: True`, say so on that line. A reader comparing a pre- and
   post-expansion best currently cannot tell whether the second number was measured.
2. **Have K's journalling state it.** `SPACE_EXPANDED` could record whether the incumbent
   remains a legal point in the widened space — if it does, a flat outcome is expected and
   the only informative comparison is among trials using new choices.

Both are small and observational. I have not applied them because the reporting layer is
where the honest-verdict logic lives and I would rather have the batch of pending decisions
resolved than add a sixth change to it. `scripts/audit_cached_bests.py` reproduces the table
above.

## A third instance, live, with the informative statistic separated out

`run-l3-43-20260905-091705`, `cand-cb7be6b4`, 10:14–10:20. K expanded five knobs
(`QKV_BLOCK_K` +256, `QKV_NUM_WARPS` +16, `ATTN_BLOCK_N` +256, `OUT_BLOCK_M` max,
`OUT_NUM_WARPS` min) and the re-tune spent 40 trials to report **14.2 ms — identical to the
pre-expansion best**, and again from the cached incumbent.

But splitting the 40 trials by whether they used a newly-added choice shows the expansion was
*not* useless, which the top-line number conceals entirely:

| | trials | complete | best | best excluding cache hits |
|---|---|---|---|---|
| used a new choice | 28 | 6 | **18.6** | **18.6** |
| only old choices | 12 | 7 | 14.2 | 19.8 |

The widened region produced a genuine **18.6 ms** measurement — worse than the 14.2 incumbent,
but *better* than the 19.8 ms best that the old choices reached under fresh measurement in this
budget. Reading only "17.1 → 17.1" or "14.2 → 14.2" would have scored this expansion as a
no-op three times over; it was not.

Note also the failure cost: 28 of 40 trials went to new-choice territory and 22 of those 28
failed, which is the real price of the expansion rather than the flat headline.

## A pre-registered prediction on the fourth instance

Written at 10:36, **before** the re-tune produced a single trial, so this is a prediction and
not a postdiction. `cand-3bf724d6`, `sp-57e98792` → `sp-f50bba31` at 10:34:55. K widened four
knobs, all purely additive:

```
QK_BLOCK_M         [16,32,64]      -> +128
QK_BLOCK_N         [16,32,64,128]  -> +256
SOFTMAX_BLOCK      [512,1024]      -> +2048
SOFTMAX_NUM_WARPS  [2,4,8]         -> +16
```

The pre-expansion θ_best (`tr-d7f2220d`, **28.6 ms**) is
`QK_BLOCK_M=64, QK_BLOCK_N=64, SOFTMAX_BLOCK=1024, SOFTMAX_NUM_WARPS=4`, and **every one of
its 13 values is still in its domain** — machine-checked, no value was removed. So the
mechanism applies exactly:

> **Prediction: if the re-tune samples that vector, `TUNING_DONE` will report 28.6 ms with
> `reused_measurement: True`, and the informative number will be the best among trials that
> used 128 / 256 / 2048 / 16.**

The only way this fails is if a genuinely new point beats 28.6, in which case the reported
best changes and the expansion was productive — which is the outcome the metric *should*
detect and the one the flat headline cannot distinguish from a cache replay. Either result is
informative; that is the point of writing it down first.

### Outcome: the prediction was WRONG — a new choice won

`TUNING_DONE` at 10:43 reported **28.0 ms, not 28.6**, from `tr-e14c7f45` with
`reused_measurement: False`. The winning vector is the incumbent with **`QK_BLOCK_M` changed
from 64 to the newly-added 128**, everything else identical. So the escape clause fired: a
genuinely new point beat the incumbent, and the reported best is a fresh measurement.

| | trials | complete | best | best uncached |
|---|---|---|---|---|
| used a new choice | 21 | 13 | **28.0** | **28.0** |
| only old choices | 19 | 17 | 28.5 | 28.5 |

**What was right:** the cache mechanism operated exactly as described — the incumbent vector
*was* resampled and *did* come back as a 28.6 cache hit (visible in the trial list, along with
two other cached replays at 32.8 and 53.6). Nothing about the caching account changes.

**What was wrong:** my prediction of the *reported best*. I had come to treat "flat re-tune"
as the near-certain outcome after three instances, and stated the alternative as a formality.
It happened on the first pre-registered test — and checking the full record afterwards showed
why it should not have been a surprise at all: **15 of 23 expansions improve** (see "The base
rate" above). I had generalized from the three cases I had investigated, which were selected
*because* they were flat, and never asked what the denominator was. That is the mistake worth
keeping: the three instances were not a sample, they were the cases that had caught my eye.

I also wrote in the first version of this section that 28.6 → 28.0 was "the first K expansion
in the record that improved a reported best". That was wrong too, and by a wide margin — it is
the sixteenth. Both errors have the same root: asserting a rate from the cases I had looked at
without counting the population, twice in one paragraph.

The improvement itself is modest: 28.6 → 28.0 is **2.1%**, sitting almost exactly on
`min_improvement_pct: 2.0`, so the widened region produced a real but marginal gain that the
convergence policy would barely count as progress.

Note the new choice is a *double-edged* one: `QK_BLOCK_M=128` produced the best trial (28.0)
and also three of the four worst (182, 201, and with `SOFTMAX_BLOCK=2048`, 203 ms). Same
pattern as `cand-cb7be6b4`'s large tiles — the fast region and the failure region are reached
by the same knob.

## Caveats

- The count is of reported *bests* that came from cache, not of cache hits overall (those are
  far more common and entirely benign — most cached trials are not the best).
- The mechanism is structural rather than statistical, and the fourth instance shows what that
  does and does not buy: **every** K expansion that leaves the incumbent legal *can* replay it
  from cache, and three of four did — but the fourth found a better point with a new choice, so
  a flat outcome is the *default*, not a law. Predicting flatness in advance failed on its
  first test.
- The load-bearing claim is therefore the negative one, which the fourth instance does not
  touch: **a flat re-tune is not evidence about the widened region**, so an expansion must be
  judged on its new-choice trials either way.
- `scripts/audit_cached_bests.py` prints the current tally; it grows as runs proceed, so the
  number in this doc is a floor rather than a total.
