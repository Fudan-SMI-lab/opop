# Consolidated fix list

Written 2026-09-08 after terminating `run-l3-43-20260908-053708` at 11.73 h. Supersedes the
ordering in `next-round-fix-plan.md` and folds in `backend-coverage-and-defects.md`. Result of the
terminated run: `docs/result-l3-43-final.md`.

Everything here is ranked by **measured cost on a real run**, with the evidence named. Items marked
DONE were fixed and verified since that plan was written; the rest are the batch to land before
L3:21 and L3:48.

---

## Status of what has already been fixed

| | what | verified how |
|---|---|---|
| DONE | `_best_profile` ranked on raw `.median`, crashing a run | L1:42 relaunch completed |
| DONE | strict timing path produced no median → silent fall back to the mean (64.8% vs 93.2% ranking accuracy) | 816/816 trials and 4/4 baselines carry medians in L3:43 |
| DONE | Tier 1 facts recorded only inside the `near_limit` block, so 4 of 5 verdict kinds lost them | 24/24 L3:43 verdicts carry occupancy, vs 1/8 pre-fix in L1:42 |
| DONE | rewrite rejections logged as `NOVELTY_REJECTED` | separate event types, both readable by `report.py` |
| DONE | only the rewriter could rescue sandbox artifacts | all three producers rescue; 1 rescue observed in L1:42 |
| DONE | `aux_output_ops` dropped at the task-cost boundary | present in L3:43's `TASK_COST_MEASURED` |
| DONE | **D1** cubin metadata unreachable for every non-Triton backend | probe on box 2: `cubin` present, was `None` |
| DONE | **D2** launched-kernel filter compared mangled against demangled names | `launched_filter: applied`, 1 kernel kept instead of 18 |

Self-calibration is also confirmed to do real work, by replaying the real `classify()` over the real
recorded evidence with old constants vs calibrated: all 25 verdicts reproduced (positive control) and
**5 of 25 flip**. Candidates at 55–77% of peak, which the guessed 0.50 would have called "saturated,
stop optimizing", are correctly `resource_limited` — and those five are the starting points of the
rewrite chains that produced the final result.

---

## The batch to land before the next task

All four change recorded numbers, so they must land **together** and **not between tasks of a
comparison set** — changing calibration moves every derived threshold.

### P1. 18% of the trial budget is spent on configs that cannot run — and the budget vanishes silently

**Evidence.** 181 of 1004 L3:43 trials failed; **180 for one reason**, Triton
`out of resource: shared memory`. Cost **0.93 h of 11.73 h (7.9%)**, ≈204 more measurements at the
16.4 s median cost of a passing trial. Failure rate 21–33% across all 16 candidates, so it is not one
bad candidate.

**Predictable without a GPU.** Every error carries `required` and `limit`, and `metadata.shared` is
populated by the Triton **compiler**. Probed on box 2 with a real pipelined `tl.dot` matmul: 6/6
configurations, compile-time figure equal to the runtime `Required:` byte-for-byte
(16384/32768/98304 fit; 147456/196608/262144 fail).

**Not expressible as a guard constraint.** `BLOCK_M*BLOCK_N*stages` has failing-min *below*
passing-max in 15 of 15 candidates — the classes overlap. Shared bytes must be read from the
compiler, never derived. (Re-confirms the already-disproved flash-attention shared formula.)

**Why it never decays.** `tpe.py:107` reports failures as `TrialState.FAIL`, and Optuna **excludes**
failed trials from the TPE model. So those 181 samples produce neither a measurement nor an avoidance
signal, and TPE keeps proposing the same doomed region.

**Fix.** A worker job that compiles at a given config and returns `metadata.shared` / `n_regs`
without launching (`kernel.warmup(...)`). Call it in `_tune` (`orchestrator.py:1017`) before
`_run_trial`; on `shared > device.max_shared_bytes_optin` record
`failure_kind="infeasible_shared_memory"` with required/limit and `tell()` it as **`PRUNED`, not
`FAIL`**, so the region enters the model. Cache by `(source_sha, params.key())`. If the probe cannot
compile, fall through to the real trial — the screen must never be what rejects a candidate.

**Gain.** ~18% more measured configs per budget, plus a TPE model no longer blind to a fifth of its
samples. Only one of this batch whose effect is visible **within** a single task.

### P2. `launch_bound` cannot fire — three independent causes, not two

**Evidence.** `cpu_issue_ms` is `None` in **848/848** trial profiles; `launch_overhead` appears in
zero events of either run. The calibrated `launch_bound_cpu_ratio = 0.8737` was derived, written into
every `bottleneck.md`, and never compared against anything.

Three causes, each sufficient on its own:

- **(a)** `measure_launch_overhead` is set only by `full_eval`, whose sole caller is the final
  re-eval — while every verdict comes from a tuning trial.
- **(b) NEW, found during the final re-eval.** The overhead block exists only in `run_eval`
  (`worker_main.py:1370`). Both L3 configs set `correctness_mode: dual_witness_relaxed`, which routes
  to `run_relaxed_correctness` (1643–2028) — a handler that mentions neither
  `measure_launch_overhead` nor `launch_overhead`. Passing the flag `True` on the configs the
  experiments actually use returns `None`, which is exactly what the re-eval measured.
- **(c)** `classify()` divides `cpu_issue_ms` by `crun.best_ms` — the L2-flushed harness latency —
  while the probe's docstring says its own warm-cache `gpu_ms` is the only valid denominator, and
  `ProfileRecord.cpu_over_gpu` honours that. Biases the ratio LOW, i.e. under-detects.

Note calibration is self-consistent by construction (`run_calibrate`'s `timed()` does no L2 flush),
so the threshold is sound and only its application was not.

**Fix.** Measure once per candidate on its tuned-best config (not per trial — that was the reason for
gating it in the first place). Add it to the relaxed handler. Give `classify()` the probe's own
denominator, separate from the `gpu_ms` used for throughput fractions.

**Then verify on a task that MUST be launch-bound** — an L1 elementwise op. Until `launch_bound` or
`overhead_floor` fires on a task chosen because it has to, "the classifier has eight classes" is a
claim about the code, not the harness. Observed so far: `compute_bound` 16, `resource_limited` 10,
`memory_bound` 7, and zero of the other five.

### P3. No fp16/bf16 compute ceiling, and it now misreads the winner

**Evidence.** **13 of 25** L3:43 verdicts carry `impossible_fraction`, all fp16/bf16 on tensor cores.
θ_best reads ~141% of the tf32 ceiling. The defect degrades **with** candidate quality: every
candidate good enough to lead is one whose headroom reading is meaningless.

Verdict KIND survives (arithmetic intensity 990 vs ridge 60 independently puts it compute-side), and
`impossible_fraction` + `disagreement` warn the agent — so it degrades loudly, it just gives the wrong
number.

**Fix.** Measure fp16 and bf16 matmul ceilings in `run_calibrate` alongside fp32/tf32, add to
`Calibration`/`DevicePeaks`, and have `classify` pick the denominator from `_candidate_precision`
(which already distinguishes fp16/bf16/tf32/ieee_fp32 — only the ceiling is missing).

### P4. The orchestrator leaves an orphaned opencode server on termination

**Evidence.** Killing the run's orchestrator left its `opencode serve` (PID 77685) reparented to
init, holding port 4096 — observed directly while terminating this run. Already known ("no SIGTERM
handler, so killing it leaves orphans") but now it has bitten during a routine stop, and the next
run's server would collide with it.

**Fix.** A SIGTERM/SIGINT handler that tears down the server and any in-flight worker before exiting.
Small, self-contained, and it makes "terminate and keep the result" — which is what was just done by
hand — a supported operation.

---

## Watch items — measure, do not change yet

**W1. Loop D has still never run on an L3 task.** `NOVELTY_ROUND_STARTED = 0` in L3:43, with 4 seed
families against `max_families_total = 3`. The gate is diagnosed and the budget raised, but the only
evidence the fix works is L1:42 (fired twice). If L3:21 and L3:48 also produce 4 seeds, one of the
paper's four loops stays unexercised on the tasks it reports.

**W2. The spill/speed inversion is now weaker.** Ranking L3:43's 13 classified candidates gave
`rho = -0.516` (faster candidates spill more). But θ_best's own compiled kernels spill **zero**
(218/122/121 regs, 0 spills) against 52 for the parent it was rewritten from. So the inversion holds
across *candidates* but not at the *optimum*, which is a further reason not to touch the spill
wording on one task's evidence. Collect the same statistic on L3:21 and L3:48.

**W3. 55% of wall clock is inside agent calls** (6.05 h of 11.73: analyst 2.61, parameterizer 1.68,
rewriter 1.57, generator 0.19). Not a defect — the loops are the method — but it bounds what any
GPU-side fix can buy, and it is how to read P1's 0.93 h: 7.9% of the total, ~20% of the GPU-side half.

---

## Deliberately NOT doing

- **D3** — `Backend = Literal["triton","cuda"]` vs the loader's `"cute"`/`"tilelang"` branch.
  Deciding the supported set is a scope decision, and adding a value to the Literal before its path
  is tested end to end is exactly how D1 and D2 came to exist.
- **Offering CUTLASS/CuTe to agents.** Needs the dependency in the worker venv (absent — any CUTLASS
  candidate is a compile error today), a contract section, and D2 (now fixed) as a precondition.
  Existing evidence says the interesting comparison is Triton vs CuTe, since handwritten CUDA lost to
  Triton at 0.874x on the same algorithm and KernelPro's 1.23x came from CuTe.
- Previously-declined items, unchanged: cross-candidate report sharing, dtype bans, early-pruning /
  greedy seed selection, `REPAIR_REVERTED`, and the change in
  `docs/finding-unreachable-correctness-gate.md`.

---

## Order

**P1 → P2 → P3 → P4**, then re-run L3:43 as a controlled comparison against
`docs/result-l3-43-final.md` (same task, same calibration change, so the comparison is clean), then
L3:21 and L3:48.

Re-running L3:43 rather than moving straight on is the point: it is the only way to attribute the
batch's effect, since P1 is the sole item whose gain shows up inside one task and 3.0126 ms is now a
fixed reference to beat.
