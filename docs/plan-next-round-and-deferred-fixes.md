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

## CORRECTION: the L3:43 K-expansion causality claim was half wrong

I reported that on `cand-6cf42e7d` all three expanded knobs took their newly-reachable extremes,
making the 6.0% gain causally attributable to the expansion. Checking the published space versions
against each other shows that is false for that candidate:

    cand-6cf42e7d  STATS_BLOCK_M  [16,32,64,128] -> [...,256,512]   NEW: 256, 512
                   APPLY_BLOCK_M  [16,32,64,128] -> [...,256,512]   NEW: 256, 512
                   GEMM_GROUP_M   not widened at all

The winner used STATS_BLOCK_M=128 and APPLY_BLOCK_M=128 -- both already in the ORIGINAL space -- and
GEMM_GROUP_M=1, on a knob the expansion did not touch. So its 6.0% came from re-tuning within the
existing space, not from any new value. I had inferred "new" from the values being at a boundary,
without reading what the original range actually was.

The other expansion does hold up:

    cand-d71b18cd  QK_BLOCK_M  [16,32,64,128] -> [...,256]   NEW: 256
                   winner uses QK_BLOCK_M = 256

That one's 4.2% is attributable, because the winning configuration uses a value that did not exist
before the expansion.

**The general lesson, which is the reusable part:** "best improved after an expansion" is not
evidence the expansion caused it -- re-tuning alone moves the number, and here it moved it 6.0%.
The check that distinguishes them is diffing the space VERSIONS and confirming the winner uses a
value from the added set. Any report of K's effect should carry that check, not just the delta.
Two of two expansions improved the best; one of two is causally attributable.

## L3:43 reached Loop C with its budget intact (a first)

At 1.802 h of 12 h, `CONVERGENCE_DECIDED` recorded global `continue` with all four families active,
and family `fam-e4df6e17` (the 5.142 ms winner) `continue` at `rewrite_rounds_used: 0`. The first
rewriter call followed.

Why this is worth marking: the two budget findings this config was meant to address are
- `run-l3-21-20260905-071312` stopped at 2.05 h of 12 h having used 2 of 6 rewrite rounds, because
  families with no correct candidate filled the active slots and ended the loop;
- 4 of 19 prior runs "converged" on 0-2 rounds, and only 1 of 19 was ever ended by wall clock.

Here all four families have a correct, tuned, classified candidate before Loop C begins, 10.2 h
remain, and `rewrite_rounds_per_family` is 5. So for the first time the structure-search claim gets
a real test rather than being cut short by the seeding phase.

What to check when the run ends: the total rounds used across families (the prior high-water mark is
low single digits), whether any family freezes as `converged` versus `budget_exhausted`, and whether
Loop D ever fires -- `max_families_total` is 6 against 4 seed families, so novelty has room for the
first time in 19 runs.

## HIGH PRIORITY, next round: calibration has no fp16/bf16 ceiling, so fp16 kernels read >100%

Found on run-l3-43-20260908-053708, candidate `cand-ec42408b` (the run's best at 4.297 ms):

    pct_of_compute_peak  107.8
    impossible_fraction  1.078
    uses_tensor_cores    true
    compute_ceiling_used tensor-core (tf32)
    COMPUTE_DTYPE        fp16      <-- from the winning trial's params

`run_calibrate` measures exactly two arithmetic ceilings, fp32 and tf32 (worker_main.py:816-825).
So an fp16 or bf16 kernel is scored against the tf32 number, and on a 4090 fp16 dense throughput is
roughly 2x tf32. The consequence is not a cosmetic overshoot -- it inverts the advice:

    denominator                achieved 95.95 TFLOP/s reads as
    tf32 88.88 (current)       107.8%  -> "compute_bound, at the ceiling, stop optimizing"
    fp16 ~160 (1.8x tf32)       60.0%  -> "40% of the ceiling still available"
    fp16 ~177.8 (2.0x tf32)     54.0%  -> "46% still available"

This is EXACTLY the bug the tf32 ceiling was introduced to fix, one precision further down. The
existing code comment states the principle -- "comparing a tensor-core kernel against the fp32
ceiling reports >100% of peak" -- and then implements it for a single tensor-core precision.

Mitigating facts, which are why this is next-round rather than mid-run:
- `impossible_fraction` fired and the `disagreement` text explicitly told the agent NOT to read it
  as "at the ceiling", naming both candidate causes. So the harness degraded loudly, as designed.
- It is driver+worker side (calibration), so it cannot reach a running experiment.
- Changing calibration shifts derived thresholds, so it must not land between tasks of one chain if
  those results are to stay comparable.

The fix, when taken: measure an fp16 (and bf16) matmul ceiling in `run_calibrate` alongside fp32 and
tf32, add the fields to `Calibration`/`DevicePeaks`, and have `classify` pick the denominator from
the candidate's precision as detected by `_candidate_precision` (orchestrator.py:105-124) rather
than from the binary tensor-core/no-tensor-core split it uses now. Note the existing
`_candidate_precision` ALREADY distinguishes fp16/bf16/tf32/ieee_fp32 -- the information is
available; only the ceiling to compare against is missing.

Watch for: whether the same >100% appears on L3:21 and L3:48. Memory records fp16 being the fastest
path on these tasks, so it should recur wherever a winner picks fp16.

## Evidence: a rewrite that worked, but NOT for the reason it stated (L3:43 H2)

`cand-ab21b44c` is the run's best at **3.752 ms** (2.93x torch_compile_tf32's 10.986 ms median). Its
stated hypothesis was "GEMM shared-memory restructure **to unblock BLOCK_N=512**": pre-cast the
c_attn/c_proj weights to the compute dtype on the host so the pipeliner stages the dominant B-tile
at 2B/element instead of 4B, dropping staged smem at BLOCK_N=512/stages=2 from 73,728B to 40,960B.

The hypothesis was implemented correctly and the blocked value did become reachable -- three trials
ran at BLOCK_N=512. **They came in at 4.826 and 4.833 ms, materially worse than the 3.752 ms winner,
which uses BLOCK_N=128.** So the stated mechanism is not why the rewrite won.

What actually paid, isolated by comparing against its parent at matched dtype:

    dtype   parent cand-6cf42e7d   H2 cand-ab21b44c   gain
    fp16    5.142 ms               3.873 ms           24.7%
    bf16    6.438 ms               3.752 ms           41.7%
    tf32    7.967 ms               5.683 ms           28.7%

It improves at EVERY precision, so this is not "bf16 happened to win" either. The weight pre-cast
halves staged-tile bytes at every block size, and that general effect is the gain; unlocking 512 was
a red herring the agent itself proposed.

Two things worth keeping from this:

1. **A rewrite's stated hypothesis is not evidence for why it worked.** Checking cost one query
   (does the winning trial use the value the hypothesis was about?) and reversed the explanation. Any
   claim of the form "feedback X drove structural change Y which produced gain Z" needs the winning
   configuration checked against X, not just Y's summary read.
2. **The loop still works when the hypothesis is wrong.** The agent proposed a specific mechanism,
   the harness measured it honestly, the wrong direction lost on latency, and the search kept the
   improvement anyway. That is the design working -- the harness never had to trust the narration.

## Answered: the rewrite rounds do run, and every one of them improved (L3:43)

One of the four questions this round was watching. At 6.3 h of 12 h, run-l3-43-20260908-053708 has
**4 FAMILY_ROUND_RECORDED events -- one per family, all evaluated, all improving**:

| family | seed best | after its round | gain |
|---|---|---|---|
| fam-6a8f0088 | 8.369 ms | **3.440 ms** | **58.9%** |
| fam-8c734843 | 14.876 ms | 7.849 ms | 47.2% |
| fam-f94ad85e | 8.108 ms | 4.620 ms | 43.0% |
| fam-e4df6e17 | 5.142 ms | 3.752 ms | 27.0% |

The 58.9% is the largest single-round improvement in the project (previous maximum 47.7%). Compare
the prior record: 4 of 19 runs "converged" having used 0-2 rounds, and only 1 of 19 was ever ended
by wall clock. Raising `rewrite_rounds_per_family` from 3 to 5 is directly responsible -- under the
old cap, combined with the active-slot behaviour that ended run-l3-21-20260905-071312 at 2.05 h, at
least two of these rounds would not have happened.

A second thing worth keeping: `fam-8c734843` is the deliberately UNFUSED family (it materializes the
(B*nh,T,T) score matrix). It improved 47.2% when rewritten -- so a structurally disadvantaged
approach still responds to the loop -- but it remains 2.3x slower than the fused leader (7.849 vs
3.440). The fusion advantage the 68.12x task-cost figure measured is structural, not something
tuning or rewriting closes.

Still open at this point: whether Loop D ever fires (novelty count is a verified 0, using
REWRITE_REJECTED to separate it from Loop C), and `final_reeval_ms` versus the 3.440 ms tuned figure.

## ESCALATED: the missing fp16 ceiling affects HALF of L3:43's verdicts, not one

I first reported this as a single verdict (cand-ec42408b at 107.8%). Auditing the whole run shows
**12 of 24 BOTTLENECK_CLASSIFIED verdicts exceed 100% of the compute ceiling**, and it gets worse as
candidates get faster -- because faster candidates are exactly the ones that went low-precision:

    candidate         ms   dtype  TFLOP/s  vs tf32   vs fp16@1.8x  @2.0x
    cand-80fea541  3.290   fp16     125.3   141.0%       78.3%     70.5%   <- the run's best
    cand-f49f5b32  3.319   fp16     124.2   139.8%       77.7%     69.9%
    cand-490a9d76  3.440   fp16     119.9   134.9%       74.9%     67.4%
    cand-f997f04c  3.567   bf16     115.6   130.1%       72.3%     65.0%
    cand-ab21b44c  3.752   bf16     109.9   123.6%       68.7%     61.8%
    cand-ec42408b  4.297   fp16      96.0   108.0%       60.0%     54.0%

All 12 are fp16 (8) or bf16 (4); all 12 report uses_tensor_cores=true. The fp16 ceiling above is
ESTIMATED from published 4090 dense figures (~2x tf32) rather than measured, because measuring it
needs an exclusive-lock matmul on a GPU currently running the experiment -- doing that would
contaminate the very timings the run is producing. Measure it properly before quoting a number.

**Why this is worse than the original write-up.** The defect does not degrade gracefully with
candidate quality -- it degrades WITH IT. Every candidate good enough to reach the top of the
leaderboard is a candidate whose compute-headroom reading is meaningless. The run's best kernel is
being told it is at 141% of the ceiling when it is plausibly at ~70-78%, i.e. roughly a quarter of
its ceiling still unused. That is the most expensive possible wrong answer, delivered to the
candidate that matters most.

**What keeps it from being a silent disaster:** impossible_fraction fires on every one of the 12,
and the disagreement text tells the agent not to read it as "at the ceiling", naming both causes. So
the agent is warned; it just is not given the right number. And the classification KIND is still
usable -- these came out compute_bound, which the analytic arithmetic-intensity check independently
agrees with (990 FLOP/byte against a ridge of 60), so "this kernel is compute-side" is sound even
though "% of peak" is not.

Priority is unchanged (next round, not mid-chain, because calibration changes shift every derived
threshold) but the scope claim in the earlier note was too small by 12x.
