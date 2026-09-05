# Measurement: improvement K expands downward too, and one of those values is meaningless

Found while checking the fourth K expansion of `run-l3-43-20260905-091705`. Two things fall out:
K's downward expansions are a real and previously undocumented third of its behaviour, and five
of them added `num_stages = 0`, which is not a valid Triton pipeline depth.

## K expands in both directions

Counting every value K has ever added to a domain, across all runs:

| direction | count |
|---|---|
| upward (above the old max) | **38** |
| downward (below the old min) | **15** |
| interior | 0 |
| new knob entirely | 0 |

Every documented K case so far (`result-row-tile-is-monotone-and-k-supplies-it.md`,
`finding-k-retune-cannot-disconfirm-its-incumbent.md`) was an upward expansion, which left the
impression that K only widens toward bigger tiles. It does not: **28% of its added values go
downward**, when the boundary optimum sits at a domain's minimum.

The live example is clean. `cand-88e76051`'s winner uses `BLOCK_N=16`, the *minimum* of
`[16, 32, 64]`, and the values above it are worse (best 21.5 at 32) or entirely infeasible
(5 of 5 fail at 64). So the boundary is at `min`, direction is `decrease`, and K added **8**. That
is the mirror image of the row-tile cases and equally correct behaviour.

### Its outcome: the headline improved, the added value lost

This is exactly the case the "judge an expansion by its new-choice trials" rule
(`finding-k-retune-cannot-disconfirm-its-incumbent.md`) was written for, and here the two readings
diverge in the direction that flatters K:

| | trials | complete | best (fresh) |
|---|---|---|---|
| **`BLOCK_N=8`** — the value K added | 6 | 5 | **22.5** |
| `BLOCK_N` in {16, 32, 64} | 34 | 18 | **19.9** |

`TUNING_DONE` reports **19.9 ms, a 2.0% improvement over 20.3** — and it is a fresh measurement,
not a cache replay. But the winner is `BLOCK_N=16` with `NUM_STAGES` moved 1 → 2; the incumbent's
own value, re-found with a different stage count. **K's added value never came close**: 22.5 at
best, 13% worse than the incumbent, with the downward direction decisively refuted.

So the top line says "+2.0%, expansion helped" and the new-choice subset says "the widened region
is unproductive". Both are true statements about different things: the re-tune found a better point
*inside the old domain*, which is a benefit of re-running the tuner rather than of widening it.
Attributing the 2.0% to K would be wrong.

This is the first case in the record where the split changes the verdict in K's *disfavour* — the
three earlier instances all had a flat headline hiding a productive new region, so I had only ever
used the rule to defend expansions. It cuts both ways, which is the point of having it.

Worth noting the 2.0% also lands exactly on `min_improvement_pct: 2.0`, so this expansion counts as
"progress" by the convergence policy on the strength of a gain its widened region did not produce.

## But five downward expansions added `NUM_STAGES = 0`

```
260904-093730  cand-8974ce1e  NUM_STAGES  [1,2,3,4] -> +0
260904-093730  cand-6d6b97b8  NUM_STAGES  [1,2,3,4] -> +0
260904-093730  cand-b9b38e21  NUM_STAGES  [1,2,3,4] -> +0
260904-093730  cand-d842feb9  NUM_STAGES  [1,2,3,4] -> +0
260904-093730  cand-c36d7820  NUM_STAGES  [1,2,3,4] -> +0
```

`num_stages` is Triton's software-pipeline depth. `1` means no pipelining; `0` is not a
meaningful depth. K extended the domain below its floor by pattern — "the optimum is at the
minimum, so try one lower" — with no notion that this particular knob has a hard floor at 1.

### What actually happened when those trials ran

Not a crash, which is why it went unnoticed. **31 trials** ran with a `*_STAGES = 0`; 29 completed
and 2 failed on correctness. The profile reports `num_stages: 0` back, so Triton accepted it —
presumably clamping or treating it as "no pipelining", i.e. a duplicate of 1.

That interpretation is supported by the latencies. Per candidate, best at 0 versus best at 1–4:

| candidate | best @ 0 | best @ 1–4 | per-value bests |
|---|---|---|---|
| `cand-8974ce1e` | 25.4 | **21.2** | `{0: 25.4, 1: 21.5, 2: 21.6, 3: 21.2, 4: 28.5}` |
| `cand-6d6b97b8` | 22.0 | **19.9** | `{0: 22.0, 1: 20.5, 2: 20.2, 3: 19.9, 4: 77.5}` |
| `cand-b9b38e21` | 18.8 | **18.8** | `{0: 18.8, 1: 18.8, 2: 19.0, 3: 19.1, 4: 28.3}` |
| `cand-d842feb9` | 19.5 | **18.9** | `{0: 19.5, 1: 19.1, 2: 19.2, 3: 18.9, 4: 22.8}` |
| `cand-c36d7820` | 21.6 | **21.0** | `{0: 21.6, 1: 21.0, 2: 21.8, 3: 21.9, 4: 22.9}` |

`0` never wins. It ties once (`cand-b9b38e21`, 18.8 both) and loses the other four times. No
reported best in any run came from a `STAGES=0` configuration — checked directly.

## So how bad is it?

**Mild.** The cost is the 31 trials themselves: at ~10 s of GPU per trial that is roughly 5
minutes, all inside one run (0904). Nothing was corrupted, no verdict changed, and TPE learned to
avoid the value on its own since it was consistently slower. The tie at 18.8 is consistent with 0
being an alias for 1 rather than something exotic.

It is worth recording for two reasons beyond the wasted trials:

1. **It shows K's boundary rule has no domain knowledge.** `NUM_WARPS: [2,4,8] → +1` appears four
   times and is *legitimate* (1 warp is valid, and `PW_WARPS=1` is in L3:21's winning θ_best). The
   same mechanism that correctly found `NUM_WARPS=1` incorrectly proposed `NUM_STAGES=0`. Only
   per-knob floors distinguish them, and nothing in the guard encodes one.
2. **Silent acceptance is the real hazard.** Had Triton rejected `num_stages=0` outright, this
   would have cost 31 `compile_error` trials and been obvious in the failure clusters. Instead it
   ran, returned plausible numbers, and reported `num_stages: 0` in the profile — so any analysis
   grouping by that field silently has a phantom sixth value that is really a duplicate of 1.

## What I would change — not applied

A minimum floor per knob kind, applied where K proposes the next value down: `NUM_STAGES >= 1`,
`BLOCK_* >= 8` or 16, `NUM_WARPS >= 1`. Small and local to K's expansion logic.

**Why I stopped:** it is a behaviour change to improvement K, whose expansion rule is already the
subject of one pending item (`n_complete_by_value` gating `at_boundary` in
`measurement-analyst-median-on-one-sample.md`), and the measured cost here is ~5 minutes in one
run. It belongs in the same batch, not in a mid-run patch.

A cheaper alternative, purely observational: have the parameterizer declare each knob's floor in
the space (it already writes `description` per domain), so K reads it rather than guessing. That
is a contract change though, which is a bigger conversation than the bug deserves.
