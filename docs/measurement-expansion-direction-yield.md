# Measurement: improvement K's real yield is 23%, and downward expansion is 0-for-8

`audit_expansion_outcomes.py` reports **67.7% improved (21 of 31)** for space expansion, and its
own output warns that this is an upper bound. This note measures the lower bound — how often the
expansion's *added values* actually produced the win — and finds a general filter worth having.

Recorded now because L3:43 `cand-546e4f4b` just spent a 4th expansion going 20.0 → 20.0, and the
run then ended at 6.24h of 12h. Expansion trials are not free budget; they are the same GPU
seconds a rewrite round needs.

## The strict test

An expansion "paid" only if the post-expansion best is better by >2% **and** that best config uses
at least one value the expansion added. Anything else is the tuner re-rolling the dice on a space
it had already searched.

| | n | share |
|---|---|---|
| gain >2% **and** best uses an added value | **7** | 23% |
| gain 0.5–2% and best uses an added value | 1 | 3% |
| no meaningful gain, or the gain came from the ORIGINAL domain | **23** | 74% |

```
total post-expansion trials spent: 1202   (411 failed, 34%)
trials per genuine win:            172
```

The seven that paid:

```
+20.0%  l3-48-0905  cand-cf0f07e7     + 7.5%  l3-43-0905  cand-2d0194cc
+15.5%  l3-43-0904  cand-d257924a     + 4.1%  l3-43-0904  cand-c36d7820
+ 7.7%  l3-43-0905  cand-45c3fd7d     + 2.7%  l3-43-0904  cand-794dfc79
                                      + 2.1%  l3-43-0905  cand-3bf724d6
```

Two of these are large enough to matter on their own (20.0% and 15.5%), so the feature is not
worthless. But 74% of expansions are a full re-tune — 40 trials — that rediscovers what the
previous space already knew. Five of the eight "flat" rows in the audit are literally cache
replays of the identical config.

## The finding that is actionable: direction predicts yield

Splitting the same data per *widened knob* by the direction that was requested:

| requested direction | knobs widened | best used an added value |
|---|---|---|
| `max` | 55 | **10 (18%)** |
| `min` | 20 | **1 (5%)** |

Restricting to runs started **after** the hard-edge filter landed (`4030458`, 09-05 05:06), so the
already-fixed `NUM_WARPS=1` / `NUM_STAGES=1` cases are excluded:

| requested direction | knobs widened | best used an added value |
|---|---|---|
| `max` | 40 | 5 (12%) |
| `min` | **9** | **0 (0%)** |

The eight post-filter downward expansions, every one a miss:

```
PW_WARPS      added [1]      OUT_NUM_WARPS added [1]      QK_NUM_WARPS  added [1]
QKV_NUM_WARPS added [2]      BLOCK_N       added [8]      PV_BLOCK_K    added [8]
QKV_BLOCK_K   added [8]      SOFTMAX_BLOCK added [512]
```

Two independent mechanisms are visible, and both are general rather than task-specific:

1. **Warp counts keep being pushed toward 1** even when the domain minimum is 2 — the hard-edge
   filter only blocks a knob whose range *already touches* the wall, so `[2,4,8]` → add `1` is
   still allowed. Three of the eight are this.
2. **`BLOCK_K`-shaped knobs get pushed to 8**, which for a `tl.dot` contraction dimension is
   illegal (`K >= 16`). `QKV_BLOCK_K` added 8 on the project's best candidate and the witness
   failed to compile, rejecting the whole expansion; `PV_BLOCK_K` did the same. The prompt was
   already strengthened for this (`finding-parameterizer-lacks-triton-pitfalls-doc.md`), and it
   still happens because `boundary_knobs_to_expand` **asks** for the downward extension before
   any prompt gets a say.

## A live instance, and it went the way the measurement predicts

`cand-fe183b2d` on the clean L3:21 rerun expanded at 16:24 and its family best improved
**22.9 → 22.2 ms (3.1%)**, `improved_family: true`. That reads as a win for improvement K and it
is not one:

```
old space best : 22.9 ms  at EXPAND_NUM_WARPS=2, PROJECT_NUM_WARPS=8
new space best : 22.2 ms  at EXPAND_NUM_WARPS=2, PROJECT_NUM_WARPS=8   <- the SAME config
the two added values, measured:
  EXPAND_NUM_WARPS=1     3 completions, best 26.7 ms   (+20% worse)
  PROJECT_NUM_WARPS=16   3 completions, best 24.8 ms   (+12% worse)
```

The identical configuration was re-measured 0.7 ms faster. Neither added value helped, and both
were measured enough times to say so. This is the 74% case in the table above, caught as it
happened rather than in replay — and it is why the strict test requires the winning config to
actually *use* an added value, rather than trusting `improved_family`.

Worth noting against my own reading: when the event came in I flagged that the gain "came from a
`min`-direction widening", which would have been a counter-example to the 0-for-8 record. Checking
the winning trial's params showed the opposite. The record is now **0-for-9** on `min`, with
`EXPAND_NUM_WARPS=1` losing by 20%.

## What I am not proposing yet

The obvious change — refuse `min`-direction expansion requests outright — is tempting at 0-for-8
and I am **not** making it on this evidence:

- **n=8 post-filter.** The single historical `min` win (`FUSED_BLOCK_N` added 16, and the best used
  it) is pre-filter, so a blanket ban rests on eight observations.
- **It would be the wrong shape of fix.** The two mechanisms above are specific and separately
  addressable: a warp-floor that checks the *domain minimum against the device* rather than
  against a literal, and a contraction-dim floor that the *directive* can respect because
  `boundary_knobs_to_expand` already has the space in hand. Both are generalizations of the
  existing `HARD_EDGE` idea rather than a new policy.
- **The 23%/172-trials figure is a cost argument, not a correctness one.** Whether that trade is
  worth it depends on what the freed trials would be spent on, which is the same
  budget-allocation question as `best_history` seeding — deferred, and correctly so.

So this note records the measurement and the two mechanisms. The concrete proposal, if it is
wanted, is to extend `HARD_EDGE` from "is the range already at the wall" to "would the requested
extension cross a wall", which subsumes both cases without naming a task, a candidate, or a knob.

Reproduce with `python scripts/audit_expansion_outcomes.py` (headline) and
`python scripts/audit_expansion_direction_yield.py` (this split).
