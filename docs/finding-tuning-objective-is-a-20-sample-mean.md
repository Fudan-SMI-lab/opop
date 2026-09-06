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

## Resolution (implemented, `a98fa62`)

The objective is now `LatencyStats.robust_ms` — the median, falling back to the mean when no
median exists (older runs, and the two timing paths that return summary statistics only).
Everything that ranks or selects uses it: `tpe.tell/best/snapshot`, the orchestrator's
per-candidate and family bests, and `stats.py`'s per-knob curves. The headline speedup stays
mean-over-mean (both sides are n=100 there, so it is already fair, and it is the more
conservative claim), with `speedups_median` published beside it so a comparison against an
externally-published median or min is possible at all.

Chosen by measurement — `scripts/probe_robust_objective.py`, 2000-sample ground truth per
config, 400 windows of n=20:

| estimator | bias | CV at n=20 | ranks a 7.6% true gap correctly |
|---|---|---|---|
| mean | ±1.7% | **24–37%** | **64.8%** (near a coin flip) |
| median | ±1.0% | **3–8%** | **93.2%** |
| trim10% | ±1.6% | 5.6–8.2% | 84.0% |
| trim20% | ±2.1% | 3.7–7.8% | 90.8% |
| min | **+9.8% to +156%** | 3.7–17% | 2.5–98% — **three of six pairs backwards** |

Median over trimmed mean because the trim fraction is a knob that has to be fitted to the
data, and the measured tail (8.9–13.2% of samples above 1.5x the median) straddles 10%: any
fixed fraction is wrong for some configurations. `min` is rejected outright — at n=20 it
reports the luckiest draw rather than the cost, and it ranked half the real config pairs
backwards. Worth stating plainly because earlier notes in this repo (including my own
comparisons above) used `min` as the informal robust reference; that was wrong.

Note the residual: even the median's CV at n=20 is 3–8%, still above `min_improvement_pct:
2.0`. This change greatly improves the search but does **not** by itself make a 2% threshold
trustworthy on this task. Left alone deliberately — changing it would confound the
cross-model comparison.

## What retaining the samples immediately revealed

The noise is **not** scattered jitter. From `tunefile-l2-37-20260907-020027`, all four
trials:

```
trial 1: [370.3, 17.5, 18.2, 18.3, 18.3, 17.4, ... 17.4]   rest spans 17.3-19.3
trial 2: [385.5, 42.0, 40.7, 40.7, 40.6, 40.0, ... 41.9]   rest spans 40.0-42.0
trial 3: [372.4, 15.5, 15.4, 15.5, 16.1, 15.5, ... 15.3]   rest spans 15.3-16.3
trial 4: [378.9, 19.5, 20.3, 20.3, 125.4, 21.5, ... 19.6]  one further mid-run stall
```

**Sample #1 is a 370–385 µs outlier every time**, and samples 2–20 are then extremely tight.
`num_warmup=3` does not absorb the first timed launch. One deterministic artifact in 20
samples was inflating every trial's mean by 1.7–2.9x. This diagnosis was impossible from
mean/std/min/max, which is the concrete argument for keeping the samples.

It also justifies the choice of median over any fixed rule: trial 4 carries a *second* stall
mid-run, where the median (20.18) is correct and even "drop the first sample" (25.67) is not.

A follow-up worth considering separately: raising `num_warmup` would remove the artifact at
its source. That is a different change — it costs GPU time on the hot path and would alter
every measurement — so it is not bundled here. The median makes the objective correct
regardless.

## One consequence for the external comparison

The other team reports baseline 90.30 µs and 11.80x → 8.37 µs on a 4090. Our `eager`
baseline is 52.40 µs (we are 1.72x faster), and this run's best trial already has
**min = 16.00 µs**. Many kernel benchmarks report min or median by default. If theirs does,
the two sets of numbers are not on the same footing. I do not know their method — this is a
hypothesis to confirm before any comparison is published, not a conclusion.
