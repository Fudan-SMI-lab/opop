# Measurement: improvement K's real yield is 23%, and downward expansion is 0-for-9

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

The nine post-filter downward expansions, every one a miss:

```
PW_WARPS         added [1]   OUT_NUM_WARPS added [1]   QK_NUM_WARPS  added [1]
QKV_NUM_WARPS    added [2]   BLOCK_N       added [8]   PV_BLOCK_K    added [8]
QKV_BLOCK_K      added [8]   SOFTMAX_BLOCK added [512]
EXPAND_NUM_WARPS added [1]   (live on the clean L3:21 rerun, lost by 20%)
```

Two independent mechanisms are visible, and both are general rather than task-specific:

1. **Warp counts keep being pushed toward 1** even when the domain minimum is 2 — the hard-edge
   filter only blocks a knob whose range *already touches* the wall, so `[2,4,8]` → add `1` is
   still allowed. Four of the nine are this, the newest being `EXPAND_NUM_WARPS` at 16:24 today.
2. **`BLOCK_K`-shaped knobs get pushed to 8**, below the `tl.dot` contraction floor
   (`K >= 16`). `QKV_BLOCK_K` added 8 on the project's best candidate and `PV_BLOCK_K` did the
   same. In both cases the first witness failed to compile, **and then the parameterizer retried
   and changed the kernel** to pad the dot to 16 with the extra lanes masked off
   (`DOT_BLOCK_K = 16 if BLOCK_K == 8 else BLOCK_K`) — invented independently by both
   candidates. So 8 does run, and loses by a wide margin (38.8ms vs 24.4 best; 57.1ms vs 14.75
   best), which is what a half-occupancy dot predicts. Either way the request is wasted: below a
   hardware wall the agent can only refuse or emulate. The prompt was already strengthened for
   this (`finding-parameterizer-lacks-triton-pitfalls-doc.md`), and it still happens because
   `boundary_knobs_to_expand` **asks** for the downward extension before any prompt gets a say.

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

The obvious change — refuse `min`-direction expansion requests outright — is tempting at 0-for-9
and I am **not** making it on this evidence:

- **n=9 post-filter.** The single historical `min` win (`FUSED_BLOCK_N` added 16, and the best used
  it) is pre-filter, so a blanket ban rests on nine observations.
- **It would be the wrong shape of fix.** The two mechanisms above are specific and separately
  addressable: a warp-floor that checks the *domain minimum against the device* rather than
  against a literal, and a contraction-dim floor that the *directive* can respect because
  `boundary_knobs_to_expand` already has the space in hand. Both are generalizations of the
  existing `HARD_EDGE` idea rather than a new policy.
- **The 23%/172-trials figure is a cost argument, not a correctness one.** Whether that trade is
  worth it depends on what the freed trials would be spent on, which is the same
  budget-allocation question as `best_history` seeding — deferred, and correctly so.

So this note records the measurement and the two mechanisms.

**Update: the concrete proposal this note ended with turned out to be inert.** It read: "extend
`HARD_EDGE` from 'is the range already at the wall' to 'would the requested extension cross a
wall', which subsumes both cases". I implemented exactly that and checked it over 6392 candidate
domains: it gives an **identical verdict in every case**, because a ladder whose next step would
cross a wall already has its edge at the wall. It cannot subsume the warp case, because
`NUM_WARPS=[2,4,8] → add 1` does not cross a wall — 1 is legal, merely slow, and that is the
tuner's business rather than a filter's. What actually fixed the contraction-dim case was adding
`("BLOCK_K","min"): 16` to the table, i.e. new data, not a new predicate. Recorded here because
the proposal is the kind that reads convincingly and measures to nothing.

## A second live instance, strictly measured: the round's first K expansion

`cand-e2cd07de` on `run-l3-21-20260905-195615` expanded `APPLY_BLOCK` and reported
`improved_family: false`, 19.4 → 19.4 ms. Strictly measured, it is the 74% case again:

```
APPLY_BLOCK=2048  n=9  best 19.4 ms      <- added
APPLY_BLOCK=4096  n=9  best 19.4 ms      <- added
overall best      19.4 ms in sp-fb745e6e at APPLY_BLOCK=1024   (the ORIGINAL domain)
winner used an added value? False
```

Two points in its favour, both worth keeping separate from the outcome:

- **Requesting it was correct on the evidence.** The pre-expansion stats gave `APPLY_BLOCK`
  `effect_pct` 7.65 with a monotone curve toward the edge (128 → 21.1 ms, 1024 → 19.6 ms). That
  is precisely the signature the filter is built to detect; the curve had simply already
  flattened at 1024, which is only knowable by measuring past the edge.
- **The expansion mechanics were clean.** Only the requested knob widened
  (`[128,256,512,1024,2048,4096]`), both constraints kept verbatim, 0 dropped, and
  `EXPANSION_CONSTRAINTS_RESTORED` fired 0 times — the prompt half held and the driver backstop
  had nothing to do. Second consecutive clean expansion since that fix.

So the sample for the strict test is now **8 of 32 (25%)** counting this one as a miss, and the
`min`-direction record is unchanged at 0-for-9 because this was a `max` request.

## The round's other two expansions, and the case against filtering harder

`run-l3-21-20260905-195615` ran three more expansions after that one. Strictly measured:

| candidate | knobs added | gain | winner used an added value | verdict |
|---|---|---|---|---|
| `cand-61759130` | BLOCK_N, BLOCK_K, NUM_WARPS (max) | 22.8 → 22.8, 0% | no | miss |
| `cand-80bf3097` | GEMM_BLOCK_N +[256,512], WARPS +[16], STAGES +[5,6], APPLY_BLOCK +[32,64] | 15.6 → **14.7**, 5.8% | **yes** (`GEMM_BLOCK_N=256`) | **PAID** |
| `cand-f66890d0` | GEMM_BLOCK_N +[256], PARTIAL_BLOCK +[2048], FINAL_NUM_STAGES +[5] | 11.1 → 11.0, 0.9% | no | miss |

Running strict tally: **9 of 35 (26%)**, essentially unchanged from the 23% headline.

**The pair in the middle is the important part, and it argues against the filter proposal.**
`GEMM_BLOCK_N += [256]` was requested on both siblings, in the same run, minutes apart, on kernels
from the same rewrite call. It **paid on one and lost on the other**:

```
cand-80bf3097  sp-274540fa   BLOCK_N=256  n=4  median 20.55  min 14.70  <- space best
cand-80bf3097                BLOCK_N=128  n=19 median 17.20  min 15.60

cand-f66890d0  sp-d36a6e38   BLOCK_N=256  n=4  median 28.65  min 17.90
cand-f66890d0                BLOCK_N=128  n=18 median 12.05  min 11.00  <- space best
```

Checked for a confound before concluding: `f66890d0`'s 256 group contains a 436 ms trial at
`WARPS=1, STAGES=1, dtype=ieee`, which would wreck any median. Holding `dtype=fp16` and
`WARPS >= 4` to remove it, 256 is *still* worse — 17.9 vs 11.0. So the verdict survives the
outlier; the outlier only exaggerates it.

Two consequences:

1. **No static filter could have got both right.** Same knob name, same direction, same task,
   same parent, adjacent in time — and opposite correct answers. The distinguishing factor is the
   kernel's own tiling (H2 fuses the BN apply into the depthwise load, changing what tile width
   is optimal), which `boundary_knobs_to_expand` cannot see. This is the strongest argument yet
   against the still-undecided proposal to refuse expansion requests on aggregate historical
   yield: the 26% figure is a *population* statistic, and the population is not homogeneous.
2. **The 26% hit rate is not obviously the right thing to raise.** The one expansion that paid
   this round produced the 14.7 ms result, which then seeded the round that reached 11.0. A filter
   tight enough to remove the two misses would have needed to distinguish them from a request that
   was, on the visible evidence, identical.

**Also worth separating: the two audits disagree here, and both are right.**
`audit_expansion_outcomes.py` scores `f66890d0`'s `GEMM_BLOCK_N += [256]` at **−88.5%** (median
with vs without) while the strict test calls the whole expansion a miss for a different reason
(the winner ignored the added values). On `80bf3097` they disagree in the *other* direction: the
median ranks 256 worst of five values while its minimum is the space best. That is the same
median-vs-minimum split recorded in `finding-latency-by-value-is-a-median.md`, showing up in the
expansion audits rather than the analyst's table. Neither script is wrong; they answer different
questions, and quoting one number without saying which question it answers would be.

Reproduce with `python scripts/audit_expansion_outcomes.py` (headline) and
`python scripts/audit_expansion_direction_yield.py` (this split).
