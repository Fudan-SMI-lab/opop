# Measurement: on two candidates the parameterizer's default beat 40 trials of tuning

`run-l3-43-20260905-091705`, `cand-fe4af2fc` (21.3) and `cand-2cda23e2` (27.4). Both reported bests
carry `reused_measurement: True` on a **first** tuning — no prior space, no incumbent — so what was
replayed is the **default witness**, which `_tune` seeds into `measured_cache`
(`orchestrator.py:443-452`). In both cases 40 trials failed to beat it.

| candidate | trials | fresh completions | best (cached default) | best fresh |
|---|---|---|---|---|
| `cand-fe4af2fc` | 40 | 13 | **21.3** | 21.4 |
| `cand-2cda23e2` | 40 | 29 | **27.4** | 28.6 |

`cand-fe4af2fc`'s nearest fresh trial differs from the default only in `FLASH_NUM_STAGES` (1 → 2)
and lands 0.1 ms behind. `cand-2cda23e2`'s best fresh is 4.4% behind over 29 completions.

## Where the default ranks, per candidate

The interesting question is whether this is luck or something systematic. Ranking each candidate's
default config among all its own completed trials:

| candidate | complete | default's rank | percentile | OOM-repaired? |
|---|---|---|---|---|
| `cand-2cda23e2` | 31 | **1** | top 3% | **yes** |
| `cand-fe4af2fc` | 15 | **1** | top 7% | **yes** |
| `cand-919059a0` | 17 | 3 | top 18% | **yes** |
| `cand-13efdcd8` | 27 | 12 | top 44% | no |
| `cand-3bf724d6` | 60 | 28 | top 47% | no |
| `cand-de802450` | 39 | 19 | top 49% | no |
| `cand-6476b4cb` | 11 | 6 | top 55% | no |
| `cand-88e76051` | 41 | 24 | top 59% | no |
| `cand-ec53c32b` | 16 | 11 | top 69% | no |
| `cand-cb7be6b4` | 38 | 27 | top 71% | no |
| `cand-2d0194cc` | 40 | **30** | top 75% | **yes** |

Non-repaired candidates cluster around the middle (44–71%, median ~55%), which is what a
reasonable-but-unoptimised default should look like. Three of the four OOM-repaired candidates sit at
the very top (3%, 7%, 18%).

## The tempting explanation, and why it is only half right

The obvious story: repair had just proved one config infeasible and chose a replacement
deliberately, so a repaired default is a *considered* config rather than a first guess — and starts
strong.

**`cand-2d0194cc` refutes the strong form of that.** It is OOM-repaired too, and its default ranks
**30th of 40 — top 75%, the worst of any candidate in the run.** So being repaired does not make a
default good.

What separates them is *what repair changed*:

| candidate | repair changed | default rank |
|---|---|---|
| `cand-2cda23e2` | `QK_NUM_STAGES` 4 → 1 | 1st |
| `cand-fe4af2fc` | `FLASH_NUM_STAGES` 3 → 1 | 1st |
| `cand-919059a0` | `ATTN_ROWS_PER_GROUP` 128 → 64 | 3rd |
| `cand-2d0194cc` | `STATS_BLOCK_M` 256 → 128 (after a wrong guess) | 30th |

The two that landed 1st both had a **stage count** cut to 1. The two that did not both had a **tile
size** halved. That is a small-n pattern (2 vs 2) and I am not claiming it generalises — but it has a
plausible mechanism: dropping `num_stages` to 1 removes pipelining, which costs little on these
memory-bound attention kernels, whereas halving a tile size directly halves the work per program and
gives up reuse. Repair is optimising for "make the witness fit", and on this hardware the cheapest
way to fit happens to also be near-optimal, while the other way is expensive.

## Why this matters beyond bookkeeping

1. **The reported number credits the wrong component.** "Tuned to 21.3, a 23.9% gain over the
   family's 28.0" reads as a tuning result. The 23.9% belongs to the *rewrite*; the 21.3 specifically
   was supplied by the parameterizer's default and merely confirmed by the tuner. Marking a cached
   best in the report (`finding-k-retune-cannot-disconfirm-its-incumbent.md`, proposal 1) covers
   this.
2. **40 trials bought nothing on these two candidates.** That is ~80 GPU-minutes across the pair for
   a 0.1 ms and a 4.4% *negative* result. Not a defect — the tuner cannot know in advance — but it
   is the clearest case yet for reporting how much a tuning actually moved its own starting point.
3. **It is weak evidence that the search space was the wrong shape.** If the default is optimal, the
   space's other 39 sampled points were all worse, which either means the parameterizer chose
   extremely well or the knobs it exposed do not matter much for this kernel. On `cand-2cda23e2`,
   29 fresh completions spanning 28.6–174.0 ms suggest the knobs matter a great deal in the *bad*
   direction and not at all in the good one.

## Not applied

Nothing here is a code change I would make mid-run. The reporting fix is already proposed elsewhere;
the per-candidate default-rank table above is a one-off analysis, and if it were worth keeping it
belongs in `scripts/audit_cached_bests.py`, which currently reports only *that* a best was cached and
not where the default sat.
