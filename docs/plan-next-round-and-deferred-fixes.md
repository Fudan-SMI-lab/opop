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


---

## DONE (2026-09-08): occupancy is absent from the evidence of a SATURATED verdict

Found on `run-l1-42-20260908-023039` (the L1:42 revalidation), by comparing what
`BOTTLENECK_CLASSIFIED` recorded against what the trial profile held.

**The asymmetry.** `classify()` records `uses_tensor_cores`, `pct_of_dram_peak`,
`compute_ceiling_used` and `ridge_flop_per_byte` unconditionally (bottleneck.py:196-228). But
`ev["occupancy"]` and `ev["occupancy_limiter"]` are written only inside the `near_limit` block at
bottleneck.py:300, and BOTH saturation verdicts return before reaching it (`memory_bound` at 255,
`compute_bound` at 261). So the moment a kernel is judged saturated, the occupancy fact vanishes
from its evidence -- while the equally-cheap, equally-Tier-1 tensor-core fact stays.

**Measured on that run.** Verdict `memory_bound` at 94.7% of the measured DRAM ceiling. The
winning trial's profile held `occupancy 0.3333, limiter "blocks_per_sm", 16/48 warp slots`, with
empty `statics_notes` -- so Tier 1 collected it correctly and the classifier simply did not carry
it. The verdict's own advice reads "Increase reuse (larger tiles, better blocking)", addressed to
a kernel that is already at the per-SM block cap.

**Why this is NOT urgent, and must not be "fixed" carelessly.** The information does reach the
agent: `analysis/compiled_kernel.md` carries the occupancy line with its own correct caveat
("THEORETICAL occupancy ... Low occupancy is not automatically bad -- a large-tile kernel can be
fastest at low occupancy"), and the analyst prompt lists that file first. So this is a
consistency and locality problem in `bottleneck.md`, not lost information -- which is exactly why
it belongs here rather than in a mid-validation patch.

**The shape of the fix**, when it is taken: move the two `ev[...]` occupancy assignments out of
the `near_limit` block so they are recorded with the other unconditional facts, WITHOUT adding
occupancy to `near_limit` for a saturated kernel. The distinction matters: a memory-bound kernel
at 94.7% of ceiling is not "limited by occupancy", and promoting it to a lever would tell the
agent to chase residency when the bytes are the wall. Record the fact; do not re-rank the advice.
Verify by rendering the L1:42 numbers through `_bottleneck_doc` and checking that the verdict
prose is unchanged while the facts section gains the line.

**Taken on 2026-09-08**, while L3:43 was running (so no validation was disturbed). The Tier 1 facts
-- occupancy, its limiter, spills, registers -- are now recorded immediately after `ev` is
initialized, before any verdict can return, so all five verdicts carry them. The lever logic is
untouched.

It was FOUR verdicts affected, not two: `overhead_floor` and `launch_bound` return even earlier
than the two saturation branches.

**One claim in the note above was wrong and is corrected here.** It said the careless version of
this fix -- dropping the `near_limit` gate's `frac_bw < dram_saturated_frac and frac_fl <
compute_saturated_frac` conditions -- would promote occupancy to a lever for a saturated kernel and
tell an agent to chase residency at 94.7% of bandwidth. That is false. Those conditions are DEAD
CODE: swept across both fractions, no input reaches the gate with either fraction above its
saturation line, because the saturation returns already took every such case. Removing them changes
no verdict. Found by applying the "careless" edit and watching the test that claimed to catch it
pass anyway -- the test was vacuous, and the assertion has been replaced with one that pins the
84.41% boundary instead (`test_the_lever_block_still_requires_an_unsaturated_kernel`).

A separate observation, NOT acted on: at 84% of the measured DRAM ceiling -- just below the
saturation line -- a kernel with 33% occupancy is classified `resource_limited` and advised to
shrink its tile. That may be the wrong advice that close to a bandwidth ceiling, but it is
pre-existing behaviour governed by the calibrated threshold, not by this change, and moving the line
is a calibration question rather than a code one.

**Confirmed load-bearing in production the same day** (run-l3-43-20260908-053708, candidate
`cand-6cf42e7d`). Verdict `compute_bound` at 84.6% of the measured tensor-core ceiling, and the
brief its analyst received now carries `occupancy = 0.0833`, `occupancy_limiter = shared_memory`,
`n_spills = 10`, `n_regs = 255`.

Why that matters concretely: `compute_bound`'s advice is "the levers are arithmetic: tensor cores /
lower precision if the accuracy gate allows, and **more independent accumulators for ILP**".
Without these four facts the agent would act on that while blind to being at 255/255 registers with
10 spills and 8% occupancy -- where adding accumulators makes the kernel strictly worse. The fix
turned advice-that-would-backfire into advice the agent can weigh against a measured constraint.

Also the first real register SPILLS in the project (8 then 10). Prior measurement on box 2 found
Triton caps registers rather than spilling (218 regs, 0 spills, 16.7% occupancy), which is why
occupancy rather than spills was called the signal that fires. On this task it does both at once,
at 97% of the shared-memory opt-in limit -- so the spill detector is not dead code after all.

## Evidence: the measured fusion headroom predicted the candidate ordering (L3:43)

run-l3-43-20260908-053708 produced an unplanned controlled comparison. `task_cost` measured the
reference at **68.12x** its compulsory traffic (416296960 B), and `_bottleneck_doc` told every agent
that fusing was the largest lever this task offers. The four seeds happened to span that axis, and
the ordering came out as the measurement predicted:

| candidate | structure | best | regs/spills | verdict |
|---|---|---|---|---|
| cand-6cf42e7d | two-pass, fused | **5.142 ms** | 255 / 6 | compute_bound 90.1% |
| cand-c8830fe8 | cuBLAS + one fused attention kernel | 8.108 ms | 255 / - | resource_limited 57.1% |
| cand-da341a61 | 3-kernel pipeline | 8.369 ms | 255 / 8 | resource_limited 55.3% |
| cand-d71b18cd | **unfused, materializes (B*nh,T,T) scores** | **15.790 ms** | 128 / 0 | - |

3.07x between the most- and least-fused candidate on one task. What makes it evidence rather than
coincidence is the loser's profile: `kernel_names` = ['_pv', '_row_softmax', '_scores'] confirms the
score matrix really is written and re-read, and at 128 registers with 0 spills it is NOT
resource-starved -- so its 15.79 ms cannot be attributed to the register pressure that limits the
faster candidates. It is paying the traffic the 68.12x figure measured.

Worth keeping for the paper: this is the task-cost measurement doing predictive work, not just
describing a result after the fact.
