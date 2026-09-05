# Measurement: the analyst's `predicted_gain_pct` overshoots 16 times in 21

Prompted by the H1 refutation (`result-analyst-hypothesis-refuted-by-control.md`), where a
quantified +8.0% prediction produced −7%. That was n=1. This is the whole record.

## The numbers

For every `BOTTLENECK_REPORTED` whose `parameter_limits` stated a positive
`predicted_gain_pct`, comparing the largest stated gain against what the parent's best child
actually achieved (`(parent_best − best_child_best) / parent_best`):

| predicted | actual | error | candidate |
|---|---|---|---|
| 30.0 | 7.6 | **−22.4** | `cand-697a898d` |
| 25.0 | 22.5 | −2.5 | `cand-cb7be6b4` |
| 25.0 | 13.4 | −11.6 | `cand-cf0f07e7` |
| 16.4 | **−16.5** | **−32.9** | `cand-372b59bd` |
| 10.6 | −0.8 | −11.4 | `cand-f4a2ce82` |
| 8.5 | 12.9 | **+4.4** | `cand-6476b4cb` |
| 8.0 | 3.8 | −4.2 | `cand-6c1ed6a8` |
| 8.0 | **−7.3** | −15.3 | `cand-13efdcd8` ← the H1 case |
| 7.0 | 0.0 | −7.0 | `cand-52c895a9` |
| 5.4 | −1.9 | −7.3 | `cand-80665a49` |
| 5.0 | 0.0 | −5.0 | `cand-f293a587` |
| 5.0 | 2.5 | −2.5 | `cand-0c3b5820` |
| 5.0 | 23.9 | **+18.9** | `cand-3bf724d6` |
| 4.0 | −5.6 | −9.6 | `cand-794dfc79` |
| 3.0 | 1.8 | −1.2 | `cand-d08506c0` |
| 3.0 | −7.1 | −10.1 | `cand-f2e09b3b` |
| 3.0 | −5.6 | −8.6 | `cand-794dfc79` |
| 3.0 | 15.0 | **+12.0** | `cand-de802450` |
| 2.6 | −0.5 | −3.1 | `cand-05f1118a` |
| 2.5 | 15.0 | **+12.5** | `cand-de802450` |
| 2.0 | 23.9 | **+21.9** | `cand-3bf724d6` |

```
n = 21   median predicted  5.0%   median actual  1.8%   median SIGNED error  -5.0%
actual >= predicted in  5 of 21
actual was NEGATIVE (child WORSE than parent) in  8 of 21
```

So the H1 case was not unusual. **Predictions overshoot 16 times in 21**, and in 8 of 21 the
rewrite that followed made the family's best *worse* rather than better by the predicted amount.

## The confound, stated plainly, because it is large

"Actual" here is not a measurement of the prediction. It is the best child's tuned latency, and:

- **A child implements a *hypothesis*, not a `parameter_limits` entry.** The two are separate
  fields of the report. A rewrite typically changes several things at once.
- **Median 2 children per parent**, so "best child" already takes a max over attempts — which
  biases *toward* the prediction, not against it.
- **The best child's approach summary mentions the predicted knob's prefix in only 10 of 22
  cases.** Less than half the time is the winning child even addressing the knob whose gain was
  predicted.

So this table measures the **whole chain** — analyst names a limit, rewriter attacks something,
tuner tunes it — and not the calibration of the number in isolation. The one case where the
chain *was* isolated is the H1 refutation: same run, same hardware, parent's own 21 trials at
the control value, verified mechanism, prediction +8.0%, outcome −7.3%. That single case is
better evidence about the number than these 21 are.

What the 21 do establish is weaker but still useful: **whatever the number means, it is not a
usable estimate of what the next round will deliver.** A reader of a `BottleneckReport` who
budgeted a round on the strength of `predicted_gain_pct: 8.0` would have been disappointed 16
times out of 21, and the two largest predictions (30.0 and 25.0) came in at 7.6 and 13.4.

## The important counterweight: *what* it names is right, only *how much* is wrong

It would be easy to read the above as "the analyst is unreliable". The evidence says something
much narrower, and the distinction matters for the paper's claim. Counting what
`parameter_limits` entries actually point at, across every report:

| knob kind named as having headroom | n | share |
|---|---|---|
| **tile size** (`BLOCK_*`, `ROW_*`, `CHUNK`) | **82** | **67%** |
| warp count | 17 | 14% |
| stage count | 14 | 11% |
| other | 10 | 8% |

Now put that beside the independent per-knob measurement of where K's widenings actually pay
(`measurement-per-knob-expansion-attribution.md`):

```
UP  TILE    16/21  76%   <- the productive case
UP  WARPS    2/9   22%
UP  STAGES   2/4   50%
```

**The analyst spends 67% of its attention on the knob class that pays 76% of the time.** Those two
numbers come from completely different derivations — one counts what the LLM chose to write about,
the other measures latency medians of trials it never saw — and they agree. That is the part of
the two-loop argument that is working: tuning statistics really do steer attention toward the
structural axis with headroom.

Its `blocked_by` attributions are also plausible rather than boilerplate:

```
registers 38   none 34   shared_memory 23   arithmetic_throughput 11   threads 9   compile_failure 8
```

34 entries say `none` — i.e. the analyst frequently declines to claim a resource is binding, which
is the honest answer and not the one that makes its report look useful.

So the finding is specifically about the **magnitude field**, not about the analyst. Direction and
target: good, and independently corroborated. Magnitude: −5.0% median bias, wrong sign 8 times in
21. Those can be true together, and treating the second as an indictment of the first would be the
error.

## Stating a number carries no signal either way

The natural follow-up: are reports that *decline* to predict less useful? No —

```
stated a gain     n=22   median actual delta +2.2%   improved 12/22
stated NO gain    n=17   median actual delta +1.8%   improved  9/17
```

Indistinguishable. An analyst that names a figure does not produce better rounds than one that
writes `predicted_gain_pct: null` and says so. `cand-ec53c32b`'s report (14:07, this run) is the
model case: it stated `null` for `PIPE_STAGES`, wrote "its gain is unproven", and gave
`NUM_WARPS` an explicit `0.0` with `blocked_by: none` — i.e. it reported the absence of headroom
rather than inventing a figure. Its round-2 children are in flight now.

That matters for what to change. The field is not load-bearing for the harness: nothing reads
`predicted_gain_pct` — `convergence.py` uses `min_improvement_pct` against measured history, and
`orchestrator.py` never consults it. It is a *reporting* field that a human reads. So the cost of
its being miscalibrated is that it misleads a reader, which is exactly the cost that matters for
a paper.

## What I would change — not applied

1. **Report the measured error alongside it.** Once a family has a child, the realised delta is
   on disk. A report line reading `predicted 8.0%, realised −7.3%` costs nothing and would have
   surfaced this on day one, the same argument as marking a cached best
   (`finding-k-retune-cannot-disconfirm-its-incumbent.md`, proposal 1).
2. **Ask for a bound rather than a point.** The analyst's own caveats are consistently the
   accurate part of its reports — `cand-13efdcd8`'s said "the gain estimate is conservative
   because the trials are unpaired", and unpaired-ness is precisely what broke it. A field like
   `evidence_strength: paired_sweep | unpaired_aggregate | single_sample` would carry more
   information than a number, and the analyst already supplies that reasoning in prose.
3. **Do not use it for allocation.** Nothing does today; this is a note to keep it that way. Any
   future ranking that weights families by predicted gain would be weighting a quantity with a
   −5.0% median bias and an 8-in-21 sign error rate.

All three are reporting-layer or prompt changes, and all three touch the analyst path, which is
already the subject of pending item 7 (`n_complete_by_value` in `ParamStat`). Adding them
piecemeal mid-run would make the batch harder to reason about, so they go on the pending list.

`scripts/audit_predicted_gains.py` reproduces both tables.
