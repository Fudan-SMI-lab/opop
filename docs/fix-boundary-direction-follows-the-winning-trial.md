# Fix: the boundary flag now points at the edge the winning trial sits on

`docs/finding-latency-by-value-is-a-median.md` established that `latency_by_value` is a
per-choice median and disagrees with the per-value minimum on 45% of knobs, then closed with an
explicit gap:

> The honest cost statement: the median can cause the analyst to *stop pushing* a knob whose
> best value it just found. Whether that has actually happened is not established here.

It has, and it costs more than analyst advice. `at_boundary` / `boundary_direction` are not
advisory: `boundary_knobs_to_expand` (`control/orchestrator.py:265`) reads them to decide which
knob improvement K extends and in which direction. A median-derived direction can therefore
spend a scarce expansion pushing away from the configuration that won.

## What made it visible

`cand-47371017` — the 9.78 ms best of `run-l3-21-20260905-195615`, and the best result of the
project so far — reported:

```
GEMM_STAGES     best_value=5   at_boundary=True  direction=max
   median table: 5 -> 13.05   4 -> 13.3   2 -> 22.5   3 -> 25.3   1 -> 25.5
   the winning trial ran GEMM_STAGES=1
```

Both numbers are true. STAGES=1 was mostly sampled alongside slow companions, and once with the
configuration that produced the run's best latency. An expansion here would have added STAGES=6.

## Measured cost

`scripts/audit_boundary_direction_vs_best_trial.py`, over 1126 knobs that have both a best
trial and published choices:

```
AGREES                                          623  (55.3%)
DISAGREES (advisory only, no boundary flag)      320  (28.4%)
MISDIRECTED (flag points away from the winner)   183  (16.3%)
```

Yield of the expansions those flags produced:

```
flag was MISDIRECTED           widened 38   best used an added value  1   (2.6%)
flag pointed at/toward winner  widened 55   best used an added value 12  (21.8%)
```

Fisher two-sided on that 2×2: **p = 0.013**.

**Confound checked.** `audit_expansion_direction_yield.py` already showed downward expansions
yield less than upward ones (5% vs 16%), so a gap explained by direction alone would be a
relabelling of a known effect. It is not — within `direction=max` alone:

```
MISDIRECTED    direction=max   widened 25   hit  0   (0%)
well-directed  direction=max   widened 53   hit 12   (23%)
```

and the misdirected flags split 98 max / 85 min, so both directions are represented.

## The change

`_param_stat` gains `best_trial_value` — the value the single fastest trial used — and anchors
the edge test on it. `best_value` still reports the median's argmin, and `latency_by_value` is
untouched, because the median remains the right *trend* statistic on a noisy laptop GPU (the
argument for it is in the earlier finding and has not changed). What changes is only which edge
the expansion machinery is told to push.

Nothing model-, task-, or knob-specific: it is a statistic swap in one predicate, and the reason
is structural — the objective is `min(trials)`, so the flag that steers search toward the
objective should be anchored to the same statistic the objective uses.

## It is mostly subtractive, which is the safe direction

The monotone-tail test still reads the median curve, deliberately: the anchor decides which edge
to *consider*, and the trend must still agree the curve heads there. Replaying all 1126 real
knobs through the new code:

```
unchanged                    874  (77.6%)
True/max  -> False/None      113  (10.0%)
True/min  -> False/None       95   (8.4%)
False/None -> True/max        24   (2.1%)
False/None -> True/min        11   (1.0%)
True/min  -> True/max          8   (0.7%)
True/max  -> True/min          1   (0.1%)
```

So 18.4% of flags are withdrawn and 3.1% newly raised. A withdrawn flag is the ambiguous case —
a winner on an edge the median slopes away from, i.e. one lucky trial against a contrary trend.
Costing an expansion *opportunity* there is preferable to spending the expansion itself at a
2.6% hit rate.

**Expansion is not starved.** Replaying the real predicate (hard-edge filter and
`min_effect_pct` included):

```
knobs REQUESTED for expansion:     old 210  ->  new 118
spaces with at least one request:  old  96  ->  new  78   (81% retained)
```

K expands at most `space_expansions_per_candidate` (=1) per candidate, so what matters is
whether a space still has *any* request. 78 of 96 do, at 1.5 requests per space instead of 2.2,
all now aimed at the winner's edge.

## Live confirmation on the next expansion, unplanned

The fix was written from the retrospective audit above. Within the hour the running L3:21 expanded
`cand-47371017`'s space and re-tuned it — a prospective test nobody arranged, on the space whose
misdirected `GEMM_STAGES` flag prompted the fix.

The old rule requested four knobs; the new rule requests two:

```
OLD requested: GEMM_BLOCK_N/max  GEMM_STAGES/max  DEPTHWISE_BLOCK/max  DEPTHWISE_WARPS/max
NEW requests : GEMM_BLOCK_N/max                                       DEPTHWISE_WARPS/max

knob                 median   winner   old flag     new flag
GEMM_BLOCK_N            128      128   True/max     True/max
GEMM_STAGES               5        1   True/max     False/None    <- withdrawn
DEPTHWISE_BLOCK        1024      512   True/max     False/None    <- withdrawn
DEPTHWISE_WARPS           8        8   True/max     True/max
```

Both withdrawn knobs are exactly the ones whose winner sat away from the median's edge. What the
re-tune then did with the values that expansion added:

```
GEMM_BLOCK_N=256   tried  0
GEMM_BLOCK_N=512   tried  1  best 17.9
GEMM_STAGES=6      tried 10  best 10.6   (2 failed)
GEMM_STAGES=7      tried  0
DEPTHWISE_BLOCK=2048  tried 4  best 12.1
DEPTHWISE_BLOCK=4096  tried 0
DEPTHWISE_WARPS=16    tried 3  best 11.1

winner: 9.42 ms, using GEMM_BLOCK_N=128, GEMM_STAGES=4, DEPTHWISE_BLOCK=512,
        DEPTHWISE_WARPS=8 -- every one of them pre-existing.
```

So the expansion did improve the candidate (9.78 → 9.42, and `GEMM_BLOCK_K=64` was already in the
space), but **by re-tuning values it already had**, not by reaching a new one. Of the 40 re-tune
trials, **13 (32%) spent themselves on values the withdrawn knobs contributed**, whose best was
10.6 ms — worse than the 9.78 ms incumbent they started from. The two knobs the new rule keeps
consumed 4 trials (10%).

This is one observation and it cannot carry the argument on its own; the 38-vs-55 audit does that.
What it adds is direction: the retrospective measurement said misdirected flags convert at 2.6%,
and the first live expansion after the fix behaved exactly that way.

### The second live expansion changes nothing, which is the point

Twenty minutes later the run expanded `cand-faa8862d` (`fam-a2688942`, 20.4 → 15.1 ms). Here the
two rules agree exactly:

```
OLD requested: BLOCK_N/max
NEW requests : BLOCK_N/max

knob             median   winner   old flag     new flag
BLOCK_N             128      128   True/max     True/max
COMPUTE_DTYPE      fp16     fp16   True/min     True/min
(six others: no flag under either rule)
```

`BLOCK_N`'s median pick and its winning value are the same, so the anchor swap is a no-op, and
`COMPUTE_DTYPE` is non-numeric so `_is_numeric_knob` filters it before direction matters. The
expansion added `BLOCK_N=256` — aimed at the edge the winner actually sits on.

Worth recording precisely because it is the un-dramatic outcome. The change is not a general
re-aiming of expansion; it is inert on the 77.6% of knobs where median and winner agree, and only
bites where they disagree. Two live expansions, one changed and one unchanged, is what a 44.7%
disagreement rate looks like at n=2.

## Propagation

`tuning/stats.py` is driver-side, so this affects the **next** run, not the one in flight —
see `opop-v2-worker-vs-driver-fix-propagation`. The results that exposed it (9.78 ms, then 9.42 ms
after the expansion) stand on their own; nothing about them is retroactively changed.

Tests: `tests/test_stats.py::test_boundary_follows_the_fastest_trial_not_the_median`,
`::test_a_winner_the_median_trend_contradicts_is_withdrawn_not_flipped`,
`::test_median_and_winner_agreeing_leaves_the_verdict_untouched`,
`::test_boundary_falls_back_to_the_median_when_the_winner_is_unmeasurable`.

Reproduce with `python scripts/audit_boundary_direction_vs_best_trial.py`.
