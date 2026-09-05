# Finding: a K re-tune cannot disconfirm its incumbent — the cache guarantees the same number

Found while trying to reproduce L3:43's 14.2 ms outlier. The measurement cache is working
exactly as designed, but it has a consequence for how improvement-K results must be read,
and it partially invalidates the way I characterized two earlier "flat" expansions.

## The mechanism

`_tune` caches measurements by parameter vector (`measured_cache`, journalled as
`reused_measurement: True` since `ea9a3a2`). A K expansion widens *some* knobs' domains and
leaves the rest alone, so the incumbent θ_best is still a legal point in the new space — and
when TPE samples it again, it is a **cache hit**, not a measurement.

Therefore: if the pre-expansion best config is resampled in the re-tune, the post-expansion
best is **arithmetically guaranteed** to be ≤ the pre-expansion best, and equal to it unless
some genuinely new point wins. A "flat" re-tune is not evidence that the widened region is
unproductive. It is the default outcome.

## Measured

Across every run, 108 reported bests. **2 came from a cached measurement**, and both are K
re-tunes on the L3:21 09-05 run:

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

## Caveats

- 2 of 108 is the count of reported bests that came from cache, not the count of cache hits
  overall (those are far more common and entirely benign — most cached trials are not the
  best).
- Both instances are on one run. The mechanism is structural rather than statistical, so the
  count matters less than the reasoning, but no other run has yet had a K expansion whose
  incumbent got resampled as the best.
