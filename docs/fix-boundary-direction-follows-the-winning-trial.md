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
bites where they disagree.

### Correction: "mostly inert" is true per KNOB and false per EXPANSION

I originally read the second and third live expansions as evidence the fix tracks the
retrospective 77.6%-unchanged rate. That conflated two units, and replaying the whole run
settles it (`scripts/audit_boundary_fix_prospective.py`):

```
run-l3-21-20260905-195615, per EXPANSION:  9 replayed, 5 CHANGED (55.6%), 4 inert
run-l3-21-20260905-195615, per KNOB:     179 verdicts, 28 changed (15.6%)
                                          of those: 24 withdrawn, 3 newly raised, 1 re-aimed
```

Both are correct and they are consistent: an expansion requests a *set* of knobs and changes if
**any** member changes, so a ~16% per-knob rate compounds across a ~3-knob set to
`1-(1-0.156)^3 = 40%`, and the observed 55.6% is in that region at n=9.

So the honest statement is two-part: **per knob** the fix is mostly inert and, where it acts,
overwhelmingly subtractive (24 of 28 changes are withdrawals, exactly 1 is a re-aiming) — but
**per expansion** roughly half of them get a different requested set. The conservatism claim rests
on the withdrawal-vs-re-aiming ratio, not on expansions being mostly untouched.

### The clearest instance so far: a small-n lucky median on the run's best candidate

The tenth live expansion is the first where the new rule empties the request list entirely
(`old=['FINAL_BLOCK'] new=[]`), and it happened on `cand-0d0dcd49`, the run's best candidate at
9.14 ms. The per-value table shows exactly why:

```
FINAL_BLOCK    n   median      min
64             8    13.55     9.33
128           14    12.00     9.23
256            5    13.60     9.47
512            6    13.45     9.14   <- the winning trial
1024           2    10.95    10.50   <- median's pick, flagged at_boundary=max
```

`1024` was sampled **twice**, caught a quiet pair, and won the median table by 1.05 ms with a
reported `effect_pct` of 24.2. Its own best trial is 10.50 — **worse than every other value's
best**. Among the top 8 trials in the space, 1024 does not appear once; 512/128/64/256 take all
eight.

The old rule therefore flagged the max edge and the run spent a parameterizer call plus a fresh
40-trial budget adding `FINAL_BLOCK=2048`, pushing further in the direction of a value whose only
evidence is two lucky samples. The new rule anchors on 512, finds it interior, and requests
nothing — the expansion does not happen at all.

This is the mechanism in its purest form, and it is also the honest counter-argument to the
"mostly subtractive is safe" framing: withdrawing this flag forgoes the *chance* that 2048 was
genuinely better. What makes that trade defensible is not certainty but the measured base rate —
misdirected flags convert at 2.6% against 21.8% — plus the observation that a 2-sample median
beating five better-sampled minima is a noise signature, not a trend.

### And then that expansion produced the run's best result, which forced a second fix

The re-tune of the space above went **9.14 → 8.13 ms (11.1%)** — the largest single expansion
gain in the project, on the space my change would have cancelled. The winning configuration used
`FINAL_BLOCK=512`, not the added 2048; every value in it was **already legal pre-expansion**, and
the pre-expansion 40 trials simply never found it.

That exposes an error in the fix as first shipped. An expansion delivers **two** things:

1. a widened range — new values the tuner could not reach;
2. a fresh tuning budget — another `trials_per_space` on the candidate.

Anchoring on the winning trial improves the aim of (1), but when it withdraws *every* request the
expansion is cancelled and (2) is forfeited as an unchosen side effect. Measured over the 43
historical expansions (`scripts/audit_expansion_cancellation_cost.py`):

```
CANCELLED (new=[])    n=8   improved 6   incl. 9.14 -> 8.13 (11.1%) and 24.00 -> 21.40 (10.8%)
RE-AIMED  (different) n=12  improved 10  -- expansion still happens, budget preserved
UNCHANGED             n=23  improved 14
```

And the counterfactual that matters: of 30 improving expansions, **16 (53%) were won by a
configuration that was already reachable** before the expansion. For those the fresh budget, not
the widened range, is what produced the gain.

**So the fix is now a floor, not a veto.** `boundary_knobs_to_expand` runs the winner-anchored
pass first; only if it returns nothing does it fall back to the median's aim. The expansion still
happens, and the only thing lost is a knob request that was a low-yield guess anyway. Cancellations
drop **8 → 1** while the re-aiming is fully preserved (5 of 10 expansions in this run still get a
different set; `cand-47371017` still goes from 4 requests to 2).

One implementation trap, caught by checking real data: `latency_by_value` is keyed in **trial
order**, not domain order — the actual stored order for `FINAL_BLOCK` is
`['128','64','256','512','1024']`. A fallback reading its first/last key as the range edges would
label 128 the minimum and mislabel directions. The fallback reads the domain's `choices` instead,
pinned by `test_median_fallback_reads_edges_from_the_domain_not_the_latency_dict`.

## Correction: the fallback is not the pre-fix rule restored, and it is wider

I described the second pass as falling back to "the median's aim", implying it reproduces
pre-fix behaviour. It does not. The two live in different places and check different things:

```
_param_stat (stats.py, the pre-fix flag)  requires THREE things:
   1. the anchor sits on the measured edge
   2. the median curve's 3-value tail is MONOTONE toward that edge
   3. the range beyond the edge is absent, or entirely failing

_median_direction (orchestrator.py, the fallback)  checks only (1).
```

Measured per knob over all 184 spaces on disk (`audit_fallback_vs_prefix_rule.py`):

```
identical verdict                                        1306
fallback flags a knob the pre-fix rule would not          219
fallback withholds one the pre-fix rule would flag          0
```

Strictly more permissive, never less. So the honest description of the shipped code is: the
first pass is the fix, and the second pass is a **wider** median rule than the one the fix
replaced — not a restoration of it.

### Why that does not change behaviour, measured rather than assumed

The per-knob predicate is not the behaviour. `boundary_knobs_to_expand` runs the fallback only
when the anchored pass returns **nothing for the whole space**, and then applies the same
numeric / `min_effect_pct` / hard-edge filters. Replaying request SETS through the real
predicate (`audit_shipped_vs_prefix_requests.py`):

```
spaces replayed                    184
identical request set              183
shipped requests MORE                1   (0 spaces gained a request from nothing)
shipped requests fewer/different     0
```

Where the 219 wider flags go:

```
204  another knob in the space already carried the anchored flag -> fallback never runs
  7  effect_pct below min_effect_pct
  8  reached the fallback
```

So the widening is almost entirely masked by the whole-space gate: a space with any
winner-anchored flag never consults the fallback, and 204 of the 219 are in exactly that
situation. **No space goes from "no request" to "some request"**, which is the property that
matters — the fallback cannot manufacture expansions the project would not have spent.

The single space where the shipped rule asks for more is `sp-cc814089` in the live L3:43 run:

```
pre-fix aim : SCORE_BLOCK_M/max  SCORE_NUM_WARPS/min  VALUE_NUM_WARPS/max
shipped     : SCORE_BLOCK_M/max  SCORE_NUM_WARPS/min  VALUE_NUM_WARPS/max
                                 + VALUE_BLOCK_M/max  + VALUE_BLOCK_D/max
```

The two extra knobs are the ones whose **winner sits on the max edge too** (`VALUE_BLOCK_M=128`,
`VALUE_BLOCK_D=128`) but whose median tail is not monotone, so the pre-fix monotone test
withheld them. Adding them is aimed at the winner, i.e. the wider rule errs in the direction the
fix argues for. That is a defensible accident rather than a designed one, and it is now recorded
as such.

### Prospective, predicted before the run acted

This was written while `sp-cc814089` had tuned once and not yet been expanded, and
`audit_fallback_prospective.py` printed the prediction. The run then emitted:

```
SPACE_EXPANDED cand-a988ff79 [SCORE_BLOCK_M/max, SCORE_NUM_WARPS/min,
                              VALUE_BLOCK_M/max, VALUE_BLOCK_D/max, VALUE_NUM_WARPS/max]
```

Exactly the predicted set, in order. The reason this is worth stating: it is the first live
exercise of the fallback path — 11 knobs, 6 on a median edge, only 2 on the winner's edge, and
`at_boundary=False` on every one, so without the fallback this expansion would have been
**cancelled** and its fresh 40-trial budget forfeited. That is the failure mode the floor was
added to prevent, observed working rather than argued for.

Reproduce with `python scripts/audit_fallback_vs_prefix_rule.py`,
`python scripts/audit_shipped_vs_prefix_requests.py`,
`python scripts/audit_fallback_prospective.py`.

## Propagation

`tuning/stats.py` is driver-side, so this affects the **next** run, not the one in flight —
see `opop-v2-worker-vs-driver-fix-propagation`. The results that exposed it (9.78 ms, then 9.42 ms
after the expansion) stand on their own; nothing about them is retroactively changed.

## First run under the fix: correctly aimed, and it still bought nothing

`run-l3-43-20260906-091019` is the first run whose driver carries this change. Its first
expansion, on `cand-8c64ccc3`, is the clean case where median and winner agree:

```
knob             median   winner   flag
BLOCK_M              64       64   True/max     <- the single requested knob
BLOCK_N              32       64   False/None
NUM_WARPS             4        4   False/None
NUM_STAGES            2        1   False/None
COMPUTE_DTYPE      bf16     fp16   False/None   (non-numeric; filtered before direction)
```

`BLOCK_M`'s anchor is the same under either rule, so the anchored pass fires directly and the
median fallback is never consulted. The expansion added `BLOCK_M=128`, aimed at the edge the
winner actually sits on — the fix working as intended.

The outcome, which is the part worth recording: **19.8 → 19.8 ms, no improvement.**

```
sp-846dfad7  BLOCK_M=16 best 22.50   =32 best 20.80   =64 best 19.80
sp-26f17306  BLOCK_M=16 best 22.80   =32 best 20.50   =64 best 19.80   =128 best 19.80
```

The added value was reached (5 trials) and tied exactly, so the widened range was genuinely
exhausted rather than unexplored, and the fresh 40-trial budget found nothing either. Correct
aim is not sufficient for a gain — it only removes one way of wasting the spend. That is
consistent with the retrospective base rate (well-directed flags convert at 21.8%, so ~4 in 5
well-aimed expansions are expected to miss), and it is a useful counterweight to reading the
9.14 → 8.13 case as what expansions normally do.

Also note `COMPUTE_DTYPE` in the second space: `effect_pct = 845.71` with the flag on the
`min` edge, i.e. fp16 is enormously faster than the alternatives and `fp16` is already the
first choice. Nothing to expand toward — the hard-edge filter is what stops that becoming a
request.

Tests: `tests/test_stats.py::test_boundary_follows_the_fastest_trial_not_the_median`,
`::test_a_winner_the_median_trend_contradicts_is_withdrawn_not_flipped`,
`::test_median_and_winner_agreeing_leaves_the_verdict_untouched`,
`::test_boundary_falls_back_to_the_median_when_the_winner_is_unmeasurable`.

Reproduce with `python scripts/audit_boundary_direction_vs_best_trial.py`.
