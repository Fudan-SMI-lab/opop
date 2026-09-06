# The tuning objective is a 20-sample mean; baselines are 100-sample means

Measured on `run-l2-37-20260907-010645` (GLM arm, level2:37 at pre-scaling sizes), first run
in the project to publish a space on the GLM arm.

## The asymmetry

| | samples | `mean/min` |
|---|---|---|
| baselines (`eager`, `eager_tf32`, `torch_compile`, `torch_compile_tf32`) | **100** | 1.04–1.15x |
| every tuning trial | **20** | **2.04x** (range 1.35–2.36x) |

Both sides are hit by the same 300–700 µs scheduling stalls. With 100 samples a handful of
outliers move the mean by 4–15%; with 20 they roughly double it. So **a candidate is
penalized against a baseline by construction**, independent of the kernel's real cost.

Path: `_run_trial` (`control/orchestrator.py:988`) calls `quick_test`, which uses
`quick_perf_trials: 20` (`evaluation/correctness.py:150-153`). Baselines and
`final_reeval` use `full_eval` → `perf_trials: 100` (`:155-158`).

## What it costs

The winning trial of the first tuning round:

```
candidate  min 16.00 | mean 32.60 | max 315.00 us   (n=20)
torch_compile_tf32 baseline  min 21.00 | mean 22.80 us   (n=100)

by mean: 32.60 / 22.80 = 0.70x   -> loses to the baseline
by min:  16.00 / 21.00 = 1.31x   -> beats the baseline
```

The same kernel gets opposite verdicts from the choice of statistic. Cf. the speed-guard
case, where one kernel got opposite verdicts depending on which side of 10x the noise fell.

Ranking damage over the 17 trials measured at the time: Spearman rho(mean-rank, min-rank)
= **0.824**. The extremes survive (tf32 32x64 is first and `ieee` last under both), but the
middle is scrambled — `bf16 64x128` is 7th by mean and 2nd by min, `fp16 128x32` is 8th and
3rd. TPE decides where to sample next from exactly that middle. The reason the ranking moves
at all is that the drag is *uneven*: `mean/min` varies from 1.21x to 2.40x across trials, a
97% spread. A uniform drag would leave the ranking intact.

## Scope, stated precisely

- **The search is corrupted; the reported result is not.** `final_reeval` uses `full_eval`
  → `perf_trials: 100` (`evaluation/benchmark.py:82-86`), the same count as the baselines,
  so the headline speedup is measured fairly. What the defect corrupts is *which point TPE
  believes is best* and which neighbourhood it explores next.
- **Not task-specific.** Every task and every run has this. It is merely loudest here: at
  ~22 µs an absolute stall of 300 µs is 14x the signal, whereas L3 tasks at 7-25 ms absorb
  the same absolute stall almost invisibly. This is why the historical L3 results remain
  trustworthy.

## The codebase already reached this conclusion once — for the reference only

`gpu/worker_main.py:855-868` fixes exactly this problem on the **reference** side:

> "The guard compares against this number, so a single scheduling stall must not decide a
> verdict. Observed live on L3:48: a 10-sample reference came back mean=609ms with
> min=29.8ms, max=5760ms, std=1720ms — one ~5.8s outlier dragged the mean 20x, producing a
> bogus 115x speedup. The MEDIAN is the robust estimator ... and keep the full distribution
> for the record."

One line earlier, at `:830-834`, the **candidate** is still summarized by `_stats_to_dict`,
which keeps only mean/std/min/max and throws the samples away (`:40-47`). Same defect, same
file, fixed on one side. And the phrase "keep the full distribution for the record" is not
actually honoured for either side — nothing persists per-sample data, so no robust statistic
can be recomputed after the fact.

## Fix (three generalized changes, not case-specific)

1. **Retain per-sample timings in the worker result** — currently impossible to recompute
   anything robust post hoc. This is open item #5, now with evidence for why it matters.
2. **Make the tuning objective robust** (trimmed mean or median) at `tuning/tpe.py:85`,
   which today passes `record.latency_ms.mean` to `study.tell`.
3. **Compare candidates and baselines at the same sample count**, or make the report state
   when they differ.

Deliberately NOT done yet: changing `quick_perf_trials` or the objective mid-flight would
make this run incomparable to the gpt arm, and the cross-model comparison is the point of
this experiment. A driver-side change cannot affect a running run anyway.

## One consequence for the external comparison

The other team reports baseline 90.30 µs and 11.80x → 8.37 µs on a 4090. Our `eager`
baseline is 52.40 µs (we are 1.72x faster), and this run's best trial already has
**min = 16.00 µs**. Many kernel benchmarks report min or median by default. If theirs does,
the two sets of numbers are not on the same footing. I do not know their method — this is a
hypothesis to confirm before any comparison is published, not a conclusion.
