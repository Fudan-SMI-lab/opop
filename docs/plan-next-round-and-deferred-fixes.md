# Deferred fixes and the next experiment round

Standing rule this file implements: fix what is **certain and low-risk** immediately; for
anything needing a large or risky change, **record it and defer** to a consolidated analysis.
Written 2026-09-07 while `run-l2-37-20260907-020707` was running.

## Deferred — high risk or large change

### D-1. The GPU lock is per-run, not machine-wide

`GpuRwLock`'s file is `store.run_dir / "jobs" / "gpu.lock"` (`wiring.py:103` →
`worker_client.py:116`). Two runs therefore create two lock files, never see each other, and
both enter the "exclusive" timing lane simultaneously. No error is raised; both runs' latency
numbers are simply wrong.

**Realized, not hypothetical**: a manual `tune-file` verification ended 02:01 and the main
experiment started 02:07 — six minutes apart, each with its own lock. Five minutes' difference
and both would have been timing at once.

*Why deferred*: the change moves the lock onto a machine-wide path, i.e. it edits the GPU
serialization path. Getting it wrong corrupts every measurement silently, which is worse than
the present state. Two call sites (`wiring.py:103`, `cli.py:76`) plus one new config field, and
it needs two concurrent processes to verify — which itself would disturb a running experiment.

*Mitigated meanwhile*: `scripts/preflight_gpu_free.py` refuses to start when another run's
event log is still growing. Read-only, and both controls are verified. It does **not** make
concurrency safe — it makes the mistake loud.

### D-2. `num_warmup=3` and the candidate-vs-baseline statistic mismatch

Candidates carry a `median` (the tuner's objective); baselines come from KernelBench's
summary-only path and have only a `mean`. So the headline comparison is across two statistics:

```
candidate median 15.10  vs  baseline mean 23.40   -> different statistics
candidate mean   32.20  vs  baseline mean 23.40   -> same statistic, candidate penalized
```

The two point opposite ways. `final_reeval` at n=100 shrinks the artifact's share from 5% to
1% but does not remove it. Raising `num_warmup` would let the mean work on both sides and make
the comparison same-statistic again.

*Why deferred*: it changes every measurement the harness produces, so it must not land
mid-round, and `scripts/probe_warmup_artifact.py` (written, **not yet run**) needs exclusive
GPU access to determine how many warmups suffice and whether the artifact is per-process,
per-compile, or per-timing-loop.

*Sizing, measured at n=158*: outliers sit at position 0 in 84.8% of cases, but also at 3, 5,
6, 16, 17. Even assuming the first sample were fully eliminated, 10.8% of trials would still
have `|mean/median − 1|` above 2% (worst 749%, from a mid-run 2898 µs stall). **So warmup is a
supplement, not a substitute for the median.** I twice called the first-sample artifact
deterministic from small samples (7/7, then 57/57) and was wrong both times — at n=158 half
the trials have a normal first sample.

### D-3. `min_improvement_pct` on a low-latency task

Left at 2.0 so the arms stay comparable. On clean trials the median's bootstrap SE is ~0.83%,
so 2.0 is workable — verified live: a 1.56% non-improvement was correctly refused. Revisit only
with cross-arm comparability in mind.

## Fixed during this round (certain and low-risk)

| Fix | Commit | Verified by |
|---|---|---|
| Nested fields sent as JSON text are decoded | `fa3e7ba` | neutralization test + two clean runs |
| Rewrite-round denominator counts families | `fa3e7ba` | all 24 runs, no impossible fractions |
| Tuning objective is a median; samples retained | `a98fa62` | live GPU run selects a different, faster kernel |
| Convergence judges on the same statistic | `b2ab4ba` | neutralization of one `update_best` call |
| `triton_pitfalls.md` #7 (`triton.lang`) and #8 (constexpr floordiv) | `f0ddb20` | a live repair cited "#7" in its diagnosis; 3 of 4 seeds hit it before, 0 of 2 rewrites after |
| Preflight refusal when another run is active | `3febd60` | positive + negative controls |
| GLM L3 token ceiling 200000 → 131072 | `3febd60` | config loads; matches the L2:37 arm |
| A median-labelled speedup requires a median on both sides | `dbcc99b` | neutralization; the mixed ratio inflated 0.727x to 1.658x |
| The deliverable trials.csv carries the median | `be3ae62` | on 280 real trials, mean-only sorting names a different winner, 44% off |

### The `triton.lang` prompt fix, measured

The one observation point that was still untested when the round began — all four seeds
predated `f0ddb20`, so nothing had exercised the new pitfalls text.

| Candidates | Generated | `triton.lang` occurrences |
|---|---|---|
| 4 seeds | before `f0ddb20` (02:51) | **3 of 4 failed to import** → 3 repair rounds spent |
| 2 rewrites | after `f0ddb20` (03:53) | **0 of 2** |

Same model, same task, same run. One repair's diagnosis text cites
"docs/triton_pitfalls.md #7" directly, which independently confirms agent-side prompt files
reach a *running* experiment (the worker-vs-driver propagation rule).

**Do not over-read it**: n=2, and the rewriter prompt is not the generator prompt, so this is
evidence rather than proof. The claim it supports is narrow — the defect that cost three
repair rounds did not recur once the pitfall was named.


## Next round, in order

### 1. gpt-5.6-sol on L3:21 — re-test whether the fixes actually help

```bash
python scripts/preflight_gpu_free.py                       # must exit 0 first
uv run kernel-opt --config configs/experiments_l3.yaml run --task level3:21
```

Config needs no change: the median objective lives in code, so this run picks it up
automatically. The comparison target is `opop-v2-l3-21-best-result` — 6.92 ms tuned / 2.18x,
whose `tuned_ms` was a 20-sample mean. Expect the re-test's `tuned_ms` to read **slower**, not
faster, because the old number was inflated by stalls; the meaningful comparison is
`final_reeval_ms` against `final_reeval_ms`.

What to check specifically:
- `speedups` (mean) vs `speedups_median`, now printed side by side.
- Whether round-over-round "improvements" that previously cleared 2.0% now fall below it. On
  L3 the effect should be much smaller than on L2:37: at 7–25 ms a 300 µs stall is minor,
  whereas at 22 µs it was 14x the signal. **If the L3 numbers barely move, that confirms the
  scope claim rather than contradicting the fix.**
- `FAMILY_ROUND_RECORDED` and the `converged` verdict, which L2:37 had not reached.

### 2. glm-5.3 on L3 tasks

```bash
uv run kernel-opt --config configs/experiments_l3_glm.yaml run --task level3:21
```

Now genuinely runnable: the double-encoding fix cleared the parameterizer (which killed
`run-l2-37-20260907-003838` outright), and the 131072 ceiling addresses the reasoning-budget
truncation that killed `run-l3-21-20260906-084636` at its first agent call.

Watch for: whether pitfalls #7/#8 stop the `triton.lang` habit in candidates generated *after*
`f0ddb20` — three of four L2:37 seeds hit it, but all three predate the prompt change, so it is
still untested.

**One run per GPU.** Run the preflight first, every time.
