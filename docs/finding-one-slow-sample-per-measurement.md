# Finding: candidate timing carries one slow sample per measurement; baselines do not

Found 2026-09-05 while checking whether `cand-a04c3f52`'s 2.46ms was real. **Status:
measured and reproduced across every candidate in the run; the mechanism is NOT yet
established.** No code change proposed until it is.

## The asymmetry

Coefficient of variation, same run, same GPU, same exclusive timing lock:

| measurement | mean | std | min | max | cv |
|---|---|---|---|---|---|
| baseline eager | 28.80 | 0.287 | 28.10 | 30.10 | **1.0%** |
| baseline eager_tf32 | 28.30 | 0.283 | 26.70 | 29.40 | **1.0%** |
| baseline torch_compile | 18.60 | 0.295 | 17.50 | 21.30 | **1.6%** |
| baseline torch_compile_tf32 | 17.90 | 0.224 | 16.90 | 19.80 | **1.3%** |
| cand-c18203b6 best | 2.09 | 1.230 | 1.80 | 7.43 | 58.9% |
| cand-f4a2ce82 best | 3.80 | 1.240 | 3.43 | 9.19 | 32.6% |
| cand-cf0f07e7 best | 2.84 | 1.800 | 2.39 | 10.70 | 63.4% |
| cand-51dd1857 best | 3.34 | 1.470 | 2.94 | 9.73 | 44.0% |
| cand-a04c3f52 best | 2.46 | 1.870 | 2.00 | 10.60 | 76.0% |

Baselines are unimodal at ~1% cv. Every candidate is bimodal at 33–76%.

## It is exactly one slow sample, not general noise

For a distribution of `n-1` samples at `min` and one at `max`, the mean would be
`((n-1)*min + max)/n`. Comparing that prediction to the observed median mean, per
candidate, over all its trials:

| candidate | trials | observed median mean | one-outlier prediction |
|---|---|---|---|
| cand-c18203b6 | 70 | 4.22 | 4.14 |
| cand-f4a2ce82 | 75 | 6.09 | 5.73 |
| cand-cf0f07e7 | 75 | 5.50 | 5.27 |
| cand-51dd1857 | 75 | 6.26 | 6.21 |
| cand-a04c3f52 | 38 | 3.18 | 3.14 |

The model fits closely for every candidate. So in essentially every 20-sample
measurement, **one sample is ~4-5x slower than the other 19** — a per-measurement fixed
cost, not run-to-run variance. Applying the same arithmetic to the baselines predicts
28.12 for eager against an observed 28.80: it does *not* fit, confirming baselines have no
such outlier.

## Why it matters

The asymmetry runs one way: it inflates candidate latency and leaves baselines alone, so
every reported speedup in every L3 run so far is **understated**. For `cand-a04c3f52`:

- as measured: 2.46ms, 7.56x vs torch_compile
- outlier-free (`min`): 2.00ms, 9.30x vs torch_compile

Two consequences beyond the headline number:

1. **The tuner optimises a contaminated objective.** TPE minimises `latency_ms.mean`. If
   the fixed cost is roughly constant across configurations, ranking survives — but the
   *fraction* it represents varies with kernel speed (76% cv at 2.46ms, 33% at 3.80ms), so
   it compresses differences between fast configurations, exactly where the search is
   trying to discriminate.
2. **It interacts with the 10x anti-hack guard.** `cand-a04c3f52` produced 11 jobs flagged
   at >=10x on the contaminated mean. On outlier-free numbers the true speedup is higher
   still, so the guard fires more often as candidates get faster — accepted and flagged
   under the current semantics, which is the right behaviour, but the flag rate is partly
   an artefact.

## What I have NOT established

**Which sample is slow, and why.** The natural guess is the first timed iteration —
Triton JIT or autotune work not absorbed by `num_warmup=3` — but `worker_main.py` keeps
only `{mean, std, min, max, n}` from `get_timing_stats`, discarding the per-sample list, so
this is not answerable from the run on disk. Alternatives not ruled out: a periodic cost
(page migration, clock/power state transition) that would not be at a fixed index, or a
cost proportional to launch count that the multi-kernel candidates pay and the single-graph
baselines do not.

The distinction decides the fix. If it is the first sample, more warmup removes it. If it
is periodic, warmup does nothing and a robust statistic (median, or trimmed mean) is the
answer. Guessing wrong would silently change every number in the paper, so:

**Next step, cheap and diagnostic:** retain the raw per-sample list in the worker result
(or its first few values plus the argmax index) for one run, then read off whether the
outlier sits at index 0. That is a small worker-side addition — it reaches a running
experiment immediately — but it should not be added mid-run, because it changes the job
output schema that the current run's readers parse.

## Interim reporting rule

Until the mechanism is known, any speedup claim from these runs should say it is computed
from contaminated means and is therefore a **lower bound**. `final_reeval_ms` uses the same
`num_warmup=3` path, so the re-eval does not escape this — the reeval-gap rule
(`docs`/memory: tuned_ms is optimistic by 1.5-6.7%) and this one are independent effects
and both apply.
