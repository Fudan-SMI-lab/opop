# Measurement: the analyst's BLOCK_N hypothesis rests on a 1-sample median

`run-l3-43-20260905-091705`, `cand-6476b4cb`, `BOTTLENECK_REPORTED` at 11:31:09, driving the
rewrite `cand-ec53c32b` at 11:34:02. Every quantitative claim in the report verifies exactly
against the trial data — and the *inference* built on them is still shakier than it reads,
because of which statistic was chosen.

## The claims all check out

| analyst said | measured | ✓ |
|---|---|---|
| "medians improve from 29.4 ms at 16 to 27.55 at 32 to 25.2 at 64" | 29.4 / 27.55 / 25.2 | exact |
| "all five BLOCK_N=128 trials fail (four runtime, one bf16 mismatch)" | 5 trials, 4 `runtime_error` + 1 `correctness_mismatch` (bf16) | exact |
| "best configuration uses 86,016/101,376 B (84.8%)" | best trial 22.5 ms, `shared_bytes` 86016 | exact |
| "the sole successful BLOCK_N=64 configuration already uses 91,136 B (89.9%)" | 25.2 ms trial, 91136 B | exact, including "sole" |
| "a lower-register fp16 config (128 registers, no spills) is slower at 24.9 ms" | present in trials | exact |

Five for five, to the digit, including a correct percentage of the opt-in limit and correct
attribution of the one bf16 failure as "unrelated". This is careful work with the data it had.

## But the medians and the bests disagree

| BLOCK_N | n complete | **median** | **best** |
|---|---|---|---|
| 16 | 4 | 29.40 | 23.4 |
| 32 | 6 | 27.55 | **22.5** |
| 64 | **1** | 25.20 | 25.2 |
| 128 | 0 | — | — |

The analyst read the median series (29.4 → 27.55 → 25.2, monotone improving) as evidence that
larger `BLOCK_N` helps and is blocked by shared memory. The **best** series says the opposite:
23.4 → 22.5 → 25.2, with the optimum at 32 and `BLOCK_N=64` *worse* than both smaller values.

And the 25.2 "median" at `BLOCK_N=64` is a **single trial** — the analyst's own text says "the
sole successful BLOCK_N=64 configuration", so it knew, and still used the value in a median
series as though the three points were comparable. A 1-sample median next to a 6-sample one is
not a trend.

Which statistic is right depends on what the question is:

- For "does this value ever produce a fast kernel?", **best** is the correct statistic, since
  tuning keeps the best and discards the rest. By that measure `BLOCK_N=64` is not blocked
  headroom; it is simply worse.
- For "is this value *typically* better?", median is defensible — but then n=1 disqualifies the
  64 point entirely, and the remaining series is 29.4 → 27.55, a 6% improvement from 16 to 32,
  which is not an argument about 128.

Either way the conclusion "BLOCK_N has 8.5% blocked headroom at 128" is not supported by these
five points. The one thing genuinely suggestive is the failure pattern: 5 of 5 at 128 fail, 8 of
9 at 64 fail, versus 9 of 13 at 16 — rising failure with tile size, consistent with a resource
wall. That is real evidence for *a* wall; it is not evidence that clearing the wall would be
worth 8.5%.

## Why this is not (necessarily) a defect

The rewrite it produced (`cand-ec53c32b`) targets shared-memory staging — "replaces the
statically unrolled key loop with a loop-local single-stage `tl.range` pipeline so only the
current K/V tile is staged, preserves the BLOCK_M=64 register-resident fp32 accumulator". If the
shared-memory wall is real, that change is useful *regardless* of whether 128 was worth 8.5%,
because it frees budget at every tile size. The hypothesis may be right for reasons better than
the statistic offered for it.

So this is filed as a measurement note, not a bug: the analyst is reasoning correctly from
sparse data, and the sparseness is the tuner's doing (8 of 9 trials at `BLOCK_N=64` failed, so
TPE could not build a sample there). The report even flags its own confounds elsewhere, as it did
on `cand-cb7be6b4` ("with cross-trial parameter confounding").

## The gap is in the harness, not the agent

I initially filed this as the analyst choosing a poor statistic. Reading `stats.py` shows that is
wrong: the agent used exactly what it was handed, and **the harness hands it only medians.**

`TuningStatsAnalyzer._param_stat` (`tuning/stats.py:42-48`):

```python
for t in complete:
    grouped[repr(t.params.values.get(name))].append(t.latency_ms.mean)
for key, vals in grouped.items():
    lat_by_value[key] = statistics.median(vals)     # <- the only per-value latency exposed
```

`ParamStat` carries `latency_by_value` (median), `failure_rate_by_value`, `best_value`,
`at_boundary`, `effect_pct` — and **no per-value sample count**. `TuningStats` has a run-level
`n_complete`, but nothing per choice. So:

- The analyst *could not* have discounted the `BLOCK_N=64` point for having n=1, because that
  fact is not in its input. It inferred n=1 anyway, from the failure rate, and said so ("the sole
  successful BLOCK_N=64 configuration") — then still used the median series, which is the only
  series it had.
- `failure_rate_by_value` does let n be *reconstructed* if the trial count per value were known,
  but it is a ratio: 8/9 failures and 80/90 give the same 0.889.

**That reframes the finding.** The agent reasoned as well as its inputs allowed and even
recovered the missing fact by inference. The defect is that a 1-sample median and a 6-sample
median are presented identically, in a field whose docstring says "Median latency per choice
(stringified choice → ms), complete trials only" with no mention of how many.

It also means the same blind spot sits in the *harness's own* boundary detector, a few lines
below: `at_boundary` walks the last three medians for monotonicity (`stats.py:83-88`) with no
regard for how many trials produced each. A single lucky or unlucky trial at the edge can set or
clear the boundary flag that drives improvement K.

## What would improve it — not applied

Add **`n_complete_by_value`** to `ParamStat`, populated from the `grouped` map `_param_stat`
already builds (`len(vals)` beside `statistics.median(vals)`), surface it in the analyst's
`tuning/stats.json`, and mention it in the input doc. That is the statistic which distinguishes
"this value is bad" from "this value was barely tried", and it costs nothing to compute because
the data is already grouped.

Two reasons to hold it anyway:

1. It feeds the hypothesis path, and prompt/schema changes to the analyst have to be judged
   against the acceptance chain downstream. Mid-run is the wrong moment.
2. The same number should probably gate `at_boundary` — a boundary set by one trial is not a
   boundary — and *that* changes when improvement K fires. Behaviour, not observability, so a
   decision rather than a fix.

Related: `finding-failure-messages-must-carry-gate-criteria.md` made the same argument for failure
details and it worked — the fix changed no verdicts and made a 190-trial pattern legible. This is
the same shape one level up, with the caveat that (2) is not merely observational.

## Outcome: the rewrite tested the hypothesis directly, and the 8.5% claim was wrong

`cand-ec53c32b` tuned to **19.6 ms** at 11:41 — a real 12.9% gain on the family (22.5 → 19.6),
so the rewrite was worth making. But it also unblocked `BLOCK_N=128`, which is exactly the
prediction under scrutiny, and the answer is unambiguous:

| BLOCK_N | parent `cand-6476b4cb` | rewrite `cand-ec53c32b` |
|---|---|---|
| 16 | 23.4 (4 complete) | 20.0 (5 complete) |
| 32 | **22.5** (6) | 19.7 (4) |
| 64 | 25.2 (1) | **19.6** (6) |
| 128 | **0 of 5 completed** | **25.9** (1 of 3 completed) |

The staging change worked mechanically — `BLOCK_N=128` went from never running to running — and
at 25.9 ms it is **32% slower** than the same kernel at `BLOCK_N=64`. The predicted "8.5% gain
from unblocking 128" is not merely unmet; the sign is wrong.

Why, visible in the profile of that one trial:

```
BLOCK_N=128:  shared 81920 (down from 90112)  regs 255  spills 642
BLOCK_N=64:   shared 90112                    regs 255  spills 2
```

The rewrite did free shared memory at the large tile — 8 KiB less — and the cost simply **moved
to registers**: 642 spills against 2. So the shared-memory wall the analyst identified was real,
and clearing it exposed a register wall immediately behind it. "Blocked by shared memory" was
true and useless: the value was blocked by *both*, and relieving one changed nothing about the
outcome.

This is the cleanest available demonstration of the point above. The bests series
(23.4 → 22.5 → 25.2, optimum at 32) predicted that 128 would be bad; the medians series
(29.4 → 27.55 → 25.2, monotone improving) predicted it would be good; **the experiment agreed
with the bests.** Had `n_complete_by_value` been in the report, the 25.2-from-one-trial point
would have carried visibly less weight and the monotone median trend would have looked like what
it was — three numbers of very unequal quality.

The rewrite still paid for itself, which is worth keeping in view: 12.9% on the family from
better staging at the tile sizes that *do* fit. A hypothesis can be wrong about its headline
number and still produce a useful change, and reporting only the 19.6 would hide both halves.

## Contrast with the register case, which was well-founded

On `cand-cb7be6b4` the analyst's register-pressure diagnosis had 15 completed trials at
`ATTN_BLOCK_M=128` versus 30 at 16, a 24% best-to-best gap, and it produced the 11.0 ms rewrite.
Same agent, same prompt, same task — the difference is entirely in how much data the tuner
managed to collect. The lesson is about *when* to trust a bottleneck report, and the answer is
visible in the trial counts, which the report does not currently carry.
