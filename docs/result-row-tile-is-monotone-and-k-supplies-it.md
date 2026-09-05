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

## The monotonicity holds across all six L3:43 runs — but 256 is NOT a fair comparison

`BLOCK_M` is the one row-tile name that recurs across runs, so it has a much larger sample
(1618 trials, all six L3:43 runs, many candidates):

| BLOCK_M | trials | fails | complete | best ms |
|---|---|---|---|---|
| 8 | 6 | 2 | 4 | 59.2 |
| 16 | 211 | 65 | 146 | 21.0 |
| 32 | 443 | 135 | 308 | 19.3 |
| 64 | 522 | 184 | 338 | 18.4 |
| **128** | 372 | 171 | 201 | **17.9** ← the best L3:43 result ever published |
| 256 | 63 | 47 (75%) | 16 | 21.7 |
| 512 | 1 | 1 | 0 | — |

Monotone down to 128 over five values and 1554 trials. **The reversal at 256 is real but it is
not evidence about the tile size**, and I initially read it as if it were. Checking which
candidates produced those 63 trials shows why:

| candidate | BLOCK_M=128 best | BLOCK_M=256 best | its design |
|---|---|---|---|
| `cand-ab6fb8ec` | 21.3 | **21.7** | "splits each head's output dim across a third grid axis **so a BLOCK_M=256 program** …" |
| `cand-913f73c9` | 21.4 | 25.0 | "partitions each **256-row** attention tile across output-dimension programs" |
| `cand-cb7a07c1` | 22.3 | 23.1 | softmax-stat pass + output-fragment passes |
| `cand-d257924a` | 28.6 | 26.2 | two-pass, low-register first pass |
| `cand-c36d7820` | 21.0 | 27.4 | shards output dim, streams Q/K in BLOCK_K chunks |
| `cand-1d133a4d` | 19.2 | 35.9 | materializes full QKV, no long-lived accumulator |
| `cand-0c3b5820` | 20.0 | — (0 of 7) | split-K local softmax partials |
| `cand-2246d8ea` | 19.7 | 25.7 (n=1) | splits output-feature ownership |

**Every candidate that reached 256 is a different program from every candidate that set the
128 record**, and most were *written specifically to make 256 feasible* — restructured to shed
registers, which costs something elsewhere. None of the fast 128 programs (`cand-794dfc79` at
17.9, `cand-b9b38e21` at 19.0) ever ran at 256 at all.

So the honest statement is: **no program has ever been measured at both 128 and 256 with
everything else held equal.** The 256 column is a comparison across *structures*, not across
tile sizes, and 7 of 8 of those structures also failed to beat their own 128 result — which is
at least consistent with the restructuring being what cost them, rather than the tile.

The monotone 16 → 128 trend has the same confound in principle, but far less severely: it is
averaged over 1554 trials and dozens of candidates, and within this run all three seeds show it
*internally*, each program compared against itself. That within-candidate evidence is what
carries the finding; the 256 row carries nothing.

Note also that 0904's published best (17.9 tuned / 19.1 re-eval) is itself a `BLOCK_M=128`
result, so the pattern is not new to today's run; today's cohort is the first where *three
different seeds* exhibit it simultaneously under three different knob names.

## The analyst reached the same conclusion from the same data, independently

`cand-cb7be6b4`'s two `BOTTLENECK_REPORTED` (10:11:53, 10:23:44) both name the row tile as the
one credible blocked knob, with the per-value progression I found:

> "The best fp16 latency by row tile progresses from 25.0 ms at 32 to 18.6 ms at 64 and 14.2 ms
> at the sampled maximum 128, **with cross-trial parameter confounding**."

It flags the confound itself, correctly identifies `registers` as the blocker (255/thread, 12
spills, with shared memory at only 64.6% and 256 of 1024 threads used), and declines to claim a
numeric gain for `QKV_BLOCK_M` because "no controlled `QKV_BLOCK_M=256` trial exists". That is
a more careful reading than my first draft of this document managed.

Its hypotheses then go **past** the 128 optimum: H1 targets `ATTN_BLOCK_M=256` by distributing a
larger query tile across more warps, and the two rewrites produced at 11:06:46 are
`cand-13efdcd8` (persistent column-local QKV CTAs serializing multiple 128-row tiles) and
`cand-919059a0` (a logical 256-row CTA built from two 128-row groups with 16 warps).

**This is the right move, not a contradiction of the 256 data** — precisely because that data
is a cross-structure comparison. Both rewrites are *new structures* designed to make 256
affordable without the register cliff, which is exactly the class of program that has never
been tested at 256 against its own 128 baseline. If either one beats `cand-cb7be6b4`'s 14.2, it
also supplies the controlled 128-vs-256 comparison the record lacks.

### Resolved: 256 is infeasible on this device, so the comparison cannot exist

`cand-13efdcd8` reached **11.0 ms** (`inprogress-l3-43-11ms-rewrite.md`) — but every one of its
seven sub-15 ms configs uses `ATTN_BLOCK_M=128`, because `QKV_M_CTAS` *serializes* 128-row tiles
rather than enlarging one.

The sibling `cand-919059a0`, which did try to widen, was rejected at its witness with
`OutOfResources: shared memory, Required: 131072, Hardware limit: 101376`
(`finding-witness-has-no-resource-precheck.md`). The arithmetic is decisive: a 256-row tile with
`D_PAD=128` needs a 256 × 128 fp32 accumulator = **131072 B** against this device's **101376 B**
opt-in limit. Repair recovered the candidate by halving the default row group, and the
republished space still *expresses* 256 (`ATTN_ROWS_PER_GROUP=64 × ATTN_ROW_GROUPS=4`) but that
point remains infeasible.

**So the controlled 128-vs-256 comparison does not exist and cannot be made on this hardware.**
That is a better answer than either of my earlier ones — better than "256 is worse" (invalid,
cross-structure) and better than "the comparison is missing" (it is impossible, not missing).
The optimum at 128 is a **hardware ceiling**, and the analyst's hypothesis was right in
substance: with 256 rows unreachable directly, distributing or serializing the work is the only
route to that much reuse, which is what produced the 11.0 ms.

### The widen candidate then confirmed the ceiling from the inside

`cand-919059a0` tuned to **14.6 ms** after repair, and its own row-tile sweep is the cleanest
version of the monotone pattern yet, because here the tile is a *product* of two knobs:

| logical rows (RPG × RG) | trials | fails | best ms |
|---|---|---|---|
| 16 | 2 | 0 | 47.3 |
| 32 | 4 | **4** | — |
| 64 | 13 | 6 | 17.1 |
| **128** | 21 | 13 | **14.6** |
| 256 | **0** | — | — |

Monotone again, optimum again at 128, and **zero trials at 256** — TPE never sampled it, because
the post-repair space carries an explicit constraint:

```
ATTN_ROWS_PER_GROUP * ATTN_ROW_GROUPS <= 128
  "Caps the fused attention row tile at the known-working default to avoid explosive
   score and output state."
```

So the 256 corner was ruled out by the *guard*, not merely left unsampled — which corrects the
paragraph above: I wrote that the tuner would "sample it, get a `runtime_error`, and learn to
avoid it". It never got the chance, and no trials were wasted at all. That is the better outcome,
but it was the constraint doing the work, not the failure feedback.

Two honest limits on this:

- **I could not determine whether that cap predates the repair.** `SPACE_REJECTED` does not carry
  the proposed space, and the parameterizer sandboxes retain only source files, not the emitted
  `space.json`. So I cannot say whether the parameterizer learned the cap from the repair's
  diagnosis or had it all along and the *default* simply violated it. The second would be a
  parameterizer bug worth its own finding; I have no evidence either way and am not guessing.
- The three seeds' monotonicity and this one's are not fully independent evidence: all four
  programs run on the same device against the same shapes, so they share whatever the true cause
  is. What they establish is that the effect is not specific to one program's structure.

Note the winner here still spills (255 regs, 12 spills, 98304 B shared = 97% of the opt-in limit)
and is 33% slower than the serializing sibling's 11.0 ms with 0 spills. On this device, the
serialize route beat the widen route decisively — which is what the register-pressure diagnosis
predicted.

## What I am not claiming

- **Not** that larger is always better, and **not** that 256 is proven worse. My first draft
  said "no seed has ever measured 256" (wrong — 63 trials exist), then said the reversal at 256
  located the optimum (also wrong — those 63 trials are a *different set of programs*, most
  written specifically to make 256 feasible, and none of the fast 128 programs was ever run at
  256). The correct statement is that **no controlled 128-vs-256 comparison exists**, which is
  what the analyst said before I did. Two errors in the same paragraph across two drafts, both
  from treating a pooled cross-candidate table as a knob sweep.
- **Not** that the parameterizer should always offer 128. On a task with different shapes the
  same choice could be uniformly bad; the evidence is that it should not stop at 64 *when the
  statistics show a boundary optimum there*, which is exactly K's existing trigger.
- **Not** that this explains the 14.2 ms. `cand-cb7be6b4` reaches 14.2 at 128 and 18.6 at 64,
  a 24% step, while the same step is 1.8% and 4.5% on the other two seeds. The tile direction is
  shared; the magnitude on that one candidate is not, and remains unexplained pending
  `final_reeval_ms`.
- **Not** that the within-run monotonicity is confound-free either. Each column pools trials
  that differ in other knobs too; what makes it stronger than the 256 row is that each column is
  one program compared against itself, over 8–15 trials per value, in three programs at once.
