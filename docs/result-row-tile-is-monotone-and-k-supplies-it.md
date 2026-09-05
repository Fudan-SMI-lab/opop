# Result: the row-tile knob is monotone on L3:43, and K keeps finding the same missing value

`run-l3-43-20260905-091705`, all four seeds tuned. Something regular appeared once the whole
cohort was on disk, and it is the clearest positive evidence for improvement K in the record —
so it is worth separating from the K-reading hazards in
`finding-k-retune-cannot-disconfirm-its-incumbent.md`.

## The regularity

Each seed parameterizes the attention row tile under a different name. Best latency per value,
across every trial of this run:

| value | `cand-cb7be6b4` `ATTN_BLOCK_M` | `cand-3bf724d6` `QK_BLOCK_M` | `cand-de802450` `BLOCK_M` |
|---|---|---|---|
| 16 | 29.8 | 29.9 | 43.0 |
| 32 | 25.0 | 29.1 | 29.4 |
| 64 | 18.6 | 28.5 | 23.5 |
| 128 | **14.2** | **28.0** | **23.4** |

**Monotone decreasing in all three, without exception**, over 8 to 15 trials per value. Three
structurally unrelated designs — a fused head-major flash kernel, a materialize-then-softmax
pipeline, and a two-pass memory-efficient pipeline — agree on the direction, and each one's
optimum is at 128. The two candidates whose domains stopped at 64 only reached that value
because improvement K added it.

## What K did with it

The knob's domain was *not* the same across seeds, and that is the interesting part:

| candidate | original domain | K's expansion |
|---|---|---|
| `cand-cb7be6b4` | `[16, 32, 64, 128]` | (128 already present; K widened other knobs) |
| `cand-3bf724d6` | `[16, 32, 64]` | **+128** → new best 28.6 → **28.0** (−2.1%) |
| `cand-de802450` | `[16, 32, 64]` | **+128** → new best 23.5 → **23.4** (−0.4%) |

Two of the three parameterizers stopped the row tile at 64. K noticed the boundary optimum in
both and added exactly the missing 128 — and **in both cases the added value took the new
best**, from a freshly measured trial:

| candidate | winning trial | change vs incumbent | cached? |
|---|---|---|---|
| `cand-3bf724d6` | `tr-e14c7f45` 28.0 | `QK_BLOCK_M` 64 → **128**, nothing else | no |
| `cand-de802450` | `tr-23227afc` 23.4 | `BLOCK_M` 64 → **128** (plus `BLOCK_N`, warps, stages, dtype) | no |

That is improvement K doing precisely what it is for: the parameterizer under-specified a
domain, the tuner hit the boundary, the statistics said "boundary optimum, direction increase",
and K supplied the value the agent had left out. **Two for two on this run**, on independent
candidates.

The honest size of the win is small, though: −2.1% and −0.4%. And `cand-de802450`'s case is
weaker than its headline, because its winning trial changed four other knobs at the same time,
so 128 is not isolated there — what *is* clean is the within-space comparison, where
`BLOCK_M=128` reached 23.4 while the old values' best fresh measurement was 23.8:

```
BLOCK_M=128 (K's addition):  13 trials,  3 complete, best 23.4
BLOCK_M in {16,32,64}:       27 trials, 13 complete, best 23.5 (cache replay) / 23.8 (fresh)
```

Note the failure cost again: the added value completed only 3 of 13 trials.

## Why this is not a cross-candidate report leak

Worth stating explicitly, because it looks adjacent to something ruled out. The user's decision
was **no cross-candidate report or hypothesis sharing** — no agent sees another candidate's
analysis. Nothing here violates that: each expansion was decided from *that candidate's own*
`TuningStats` boundary flag, independently, by the harness. The cross-candidate pattern is
visible only to me, reading the events afterwards; it was never an input to any agent or to K.

## The caution that comes with it

Large tiles are also where the failures live, which
`inprogress-l3-43-14ms-outlier.md` already recorded for `cand-cb7be6b4`:

| candidate | fail rate at 16 | at 128 |
|---|---|---|
| `cand-cb7be6b4` | 11/30 = 37% | 10/15 = 67% |
| `cand-3bf724d6` | 2/11 = 18% | 6/13 = 46% |
| `cand-de802450` | 5/12 = 42% | 10/13 = 77% |

So the direction is reliable and the *reachability* is not. Every candidate pays a rising
failure rate for the region that holds its best result, which is why a 40-trial budget lands
the large-tile optimum only sometimes — and why `cand-cb7be6b4`'s 14.2 has n=1.

## Where this sits in K's overall record

With both re-tunes done, the full tally (`scripts/audit_expansion_outcomes.py`) is **24
expansions: 16 improved, 6 flat, 2 worse**. These two are the 15th and 16th improvements, and
they are the two whose *mechanism* is fully traceable — a specific under-specified domain, a
specific boundary flag, a specific added value that then won a fresh measurement. That makes
them the best evidence for K in the record even though their magnitudes (−2.1%, −0.4%) are near
the bottom of the improvement range.

## The monotonicity holds across all six L3:43 runs — and it stops at 128

`BLOCK_M` is the one row-tile name that recurs across runs, so it has a much larger sample
(1618 trials, all six L3:43 runs, many candidates):

| BLOCK_M | trials | fails | complete | best ms |
|---|---|---|---|---|
| 8 | 6 | 2 | 4 | 59.2 |
| 16 | 211 | 65 | 146 | 21.0 |
| 32 | 443 | 135 | 308 | 19.3 |
| 64 | 522 | 184 | 338 | 18.4 |
| **128** | 372 | 171 | 201 | **17.9** ← the best L3:43 result ever published |
| 256 | 63 | **47 (75%)** | 16 | 21.7 |
| 512 | 1 | 1 | 0 | — |

Monotone down to 128 over five values and 1554 trials, then **it reverses at 256**. So the
regularity is real and it has a located optimum rather than an open direction — which is the
answer to the extrapolation question rather than a guess about it.

Note also that 0904's published best (17.9 tuned / 19.1 re-eval) is itself a `BLOCK_M=128`
result, so the pattern is not new to today's run; today's cohort is the first where *three
different seeds* exhibit it simultaneously under three different knob names.

## What I am not claiming

- **Not** that larger is always better. **Corrected:** I first wrote that "no seed has ever
  measured 256 on this task" and used that to decline extrapolating. That was wrong — 256 has
  63 trials and 512 has one, across earlier runs. The data is better than my caution: 256 is
  measurably *worse* (best 21.7 vs 17.9) and fails 75% of the time. The optimum is at 128 and
  the monotone run ends there.
- **Not** that the parameterizer should always offer 128. On a task with different shapes the
  same choice could be uniformly bad; the evidence is that it should not stop at 64 *when the
  statistics show a boundary optimum there*, which is exactly K's existing trigger.
- **Not** that this explains the 14.2 ms. `cand-cb7be6b4` reaches 14.2 at 128 and 18.6 at 64,
  a 24% step, while the same step is 1.8% and 4.5% on the other two seeds. The tile direction is
  shared; the magnitude on that one candidate is not, and remains unexplained pending
  `final_reeval_ms`.
