# Deferred findings — recorded during the L3 runs, to be handled together afterwards

Each entry states the evidence, why it did **not** meet the bar for pausing a running
experiment, and what a fix would have to change. The bar (from the runbook): a defect is
pause-worthy only if it materially damages the RESULT and the fix is clear and low-risk.
A budget-efficiency loss is not a wrong result.

Nothing here should be implemented while an experiment is in flight — several of these change
search behaviour, which would make the runs before and after incomparable.

---

## D1. A categorical knob value that never succeeds keeps being drawn

**Evidence.** `scripts/audit_dead_knob_values.py`, on box 1's L3 runs:

| run | trials | hopeless values | trials spent on them |
|---|---|---|---|
| run-l3-21-20260908-232211 | 40 | `COMPUTE_DTYPE=bf16` (8 drawn, 0 completed) | 5 (12%) |
| run-l3-48-20260907-202457 | 720 | 10 values across 5 candidates, incl. `COMPUTE_DTYPE=bf16` 27 drawn / 0 completed | 99 (14%) |
| **total** | **960** | | **104 (10.8%)** |

On L3:21 the bf16 draws were trials 5, 6, 10, 12, 17, 26, 32, 35 — spread across the whole run,
so there is no learning curve at all. Verified in `tpe.py:119`: `correctness_mismatch` is
reported as `TrialState.FAIL`, and Optuna excludes FAIL from the TPE model entirely (measured
elsewhere: 12 FAIL trials leave 1 visible to the sampler; the same 12 as PRUNED leave 13).

**Why the current behaviour is deliberate, and not simply wrong.** The code's own comment gives
the reason: a `correctness_mismatch` can be non-deterministic, or a defect in the candidate
rather than a property of the point, so teaching the sampler to avoid that region risks teaching
it noise. That argument is sound for a scattered failure. It does not cover a categorical value
that fails on every draw, where the determinism is visible in the data.

**Why not now.** This costs trial budget; it does not corrupt a result. The best candidate is
still selected from trials that passed, and the correctness gate is unaffected. Any fix changes
which points the sampler visits, so a run started after it is not comparable to one before —
precisely what must not happen mid-experiment.

**What a fix must do.** Escalate a *value* from FAIL to PRUNED only after it has failed N times
with zero successes, per (candidate, knob, value) — never pooled across candidates, since a
value fatal for one candidate is fine for another (L3:48: `COMPUTE_DTYPE=tf32` is hopeless for
`cand-2ba9d5ea` and `cand-90060cab`, while tf32 completes 16 of 17 on L3:21's candidate). It
must not touch the first N draws, so a genuinely flaky failure still gets explored. N=3 is what
the audit uses and is a starting point, not a measured optimum.

**Not to be confused with a dtype ban**, which is explicitly out of scope by prior decision.
This does not remove a choice from the space; it stops re-drawing one the run has already
disproved for that candidate.

---

## D2. bf16 is 0-for-22 on this task, across two candidates and two knob names

**Corrected 2026-09-09.** An earlier version of this entry was titled "`STORE_DTYPE=fp16` is a
genuine defect the space still offers" and concluded that storing intermediates in fp16 destroys
this task's result. **That was wrong**, and it was wrong because I read a 40-trial slice as if it
were the run. With 149 trials the same knob is 36 complete / 22 failed, and the run's **best
trial uses `STORE_DTYPE=fp16`** (5.2741 ms, with `COMPUTE_DTYPE=fp16`). The 1184x failures I
attributed to the store dtype belong to the combinations that also set bf16 or ieee compute; fp16
store is not the variable that separates them.

**What the fuller data actually shows.** L3:21 at 149 trials, per (knob, value) across both
candidates:

| knob | value | complete | failed |
|---|---|---|---|
| COMPUTE_DTYPE | **bf16** | **0** | **14** |
| COMPUTE_DTYPE | fp16 | 27 | 7 |
| COMPUTE_DTYPE | tf32 | 21 | 3 |
| COMPUTE_DTYPE | ieee | 5 | 3 |
| GEMM_PRECISION | **bf16** | **0** | **8** |
| GEMM_PRECISION | fp16 | 8 | 4 |
| GEMM_PRECISION | tf32 | 32 | 14 |
| GEMM_PRECISION | ieee | 8 | 2 |
| STORE_DTYPE | fp16 | 36 | 22 |
| STORE_DTYPE | fp32 | 17 | 5 |

`COMPUTE_DTYPE` and `GEMM_PRECISION` are the **same knob independently named by two different
candidates** (cand-2d4e1574 and cand-37572704). bf16 is 0-for-14 under one name and 0-for-8 under
the other: **22 failures, zero successes, two agents, one task.** Every other value passes
somewhere. That pattern points at the task, not at a candidate defect — which is the opposite of
what the earlier entry concluded.

Failures are also spread rather than concentrated: 21 mismatches on one candidate and 28 on the
other, so this is not one broken candidate dragging the rate up.

**The gate is behaving correctly.** Worth stating explicitly, because the opposite failure — a
correct kernel rejected by an over-tight gate — has happened on other tasks and would otherwise
be the assumed diagnosis. `bf16/fp32` fails at `ratio_to_reference: 8.000` against a 3.0
multiplier, and the reference's own ieee-vs-tf32 noise floor is reported alongside every rejection
(`frac 0.955`, `cosine 0.99999975`). The candidates are 8x worse than the reference's own distance
from an fp64 golden. Nothing needs loosening.

**Why not now.** MBConv in train mode normalizes with batch statistics, so the reduction runs over
values whose magnitudes bf16's 8-bit mantissa cannot hold — a plausible mechanism, but I have not
demonstrated it, and a task-specific numerical claim is exactly what must not be acted on
mid-experiment. The cost is the budget waste counted in D1.

**What a fix would need first.** A demonstration, not this inference: compute the reference's
batch statistics in bf16 against fp64 on this task's real shapes and show the error exceeds the
gate. Only then is a prompt sentence about bf16's mantissa in normalization reductions justified,
and prompt edits reach a running experiment mid-flight (the `.md` files are re-read per call), so
it waits regardless.

---

## D3. The rewriter's backend switch is unverified in production

**Corrected 2026-09-09.** This entry originally claimed `1ef142d` landed "after
`run-l3-21-20260908-232211` started", making L3:21 a backend-switch control. **That was wrong.**
Box 1's reflog shows the checkout fast-forwarded to `e2d3d32` at 23:19:34 and the orchestrator
started 23:22:10; `1ef142d` (committed 23:03:08) is an ancestor of `e2d3d32` (23:08:05).
Verified in the loaded source on the box: the running `orchestrator.py` registers rewrites via
`_detect_backend`, and the running `modules.py` carries the backend-switch prompt section.
**L3:21 had the backend switch for its entire duration.** Only the ceilings fix (`9e8066d`,
23:59:24) postdates it. The lesson is the usual one: a run's capabilities are decided by the
box's reflog and the loaded source, not by commit-message chronology recalled from memory.

**Evidence so far (both runs combined).** L3:21: 6 rewrites; L3:43 (in flight, at 6 rewrites):
6 rewrites. **12 of 12 declared and detected Triton, 0 `BACKEND_DECLARATION_MISMATCH`, 0
CUDA-declared candidates.** The declaration/detection machinery agrees with itself; no rewrite
has yet taken the CUDA option.

**Why not now.** Nothing is broken; the option is expressible, the prompt names when a switch is
warranted, and the agents have not chosen it. **Zero CUDA candidates is a legitimate result to
report**, not a failure — 35 of 35 Triton candidates in earlier runs was a consequence of what
the prompts asked for; in these two runs the alternative was available and declined. One caveat
for the writeup: both tasks so far (MBConv, causal attention) are ones where the prompt's own
stated trigger — strict IEEE fp32 dot-bound work — does not dominate, so "declined" is weak
evidence about the trigger case itself. L3:48 runs strict-precision GEMM-heavy work and is the
first real test of the trigger.

---

## D4. The measured ceilings' effect on agent behaviour has no clean experiment

**Evidence.** Routing the ceilings into `device.md` landed in `9e8066d`, also after L3:21
started. Separately measured (`scripts/audit_ceiling_omission_impact.py`): without the ceilings,
11 of 12 seeds still declared a precision knob and 23 of 24 published spaces still exposed a
precision domain, and the one seed without a knob has no `tl.dot` at all — so the omission cost
information, not behaviour.

**Why not now.** The fix is in and is an improvement, not a repair. The open question is whether
better-informed precision choices produce better candidates, and that cannot be answered by
comparing L3:21 (no ceilings) against L3:43 (ceilings) because the task differs.

**What a clean answer needs.** The same task run both ways. That is a next-round experiment with
its own budget, not a fix.

---

## D5. F5's compile prescreen is net NEGATIVE on L3:21

**Found because it tripped a stall alarm.** A prescreen worker ran 15 minutes on a 40-variant
batch, long enough for `run_progress.py` to report SUSPICIOUS. It was not wedged — `ptxas` was
actively compiling and the worker's CPU time advanced 56 s → 69 s across a 45 s sample — but the
duration itself was the finding.

**Measured** (`scripts/audit_prescreen_cost.py`, on run-l3-21-20260908-232211 at 6.3 h):

| | |
|---|---|
| prescreens | 12 (two per candidate: one per published space, and each space expansion publishes a new one) |
| wall clock in prescreen | **27.0 min** |
| per prescreen | 76–260 s (median ~123 s) |
| configs the sampler avoided | 56 |
| trial time avoided (56 × 18.6 s) | 17.4 min |
| **net** | **−9.6 min** |

Against the design measurement — 16.7 s process start plus ~7 ms marginal per config, 11.02 s for
48 configs — the real cost is **7 to 24× higher**. The design figure was taken on ONE simple
pipelined matmul; an L3:21 candidate carries several kernels per variant (a projection GEMM, an
expand GEMM, a depthwise kernel), and every kernel of every variant compiles separately. The
marginal cost is per *kernel*, not per variant, and the design measurement missed that.

**An arithmetic correction I made in the process.** My first version of the audit reported
**+0.6 min** (break-even) by adding the 56 guard-level avoidances to the 33
`CONFIG_SCREENED_INFEASIBLE` refusals. That double-counts: all 33 of this run's
`infeasible_shared_memory` trials carry `"compile-only screen"` in their detail, so the
post-materialize refusals ARE those trials. And a post-materialize refusal is not a saved trial
at all — it still consumed a trial slot and a worker round-trip, and only avoided a launch that
would have raised. Only the guard-level count removes a point from the sampler entirely.

**The screen's correctness half is working.** All 33 shared-memory trials were refused by the
compiler's own figure rather than reaching a launch, so nothing raised `out of resource`. On
`run-l3-43-20260908-053708` the same class of failure cost 180 of 1004 trials. The screen is not
broken; it is priced wrong on multi-kernel candidates.

**Why not now.** 9.6 min lost against 6.3 h is 2.5% — an efficiency loss, not a wrong result, and
the best trial is 3.6050 ms against a 6.92 ms incumbent. Changing the screen mid-run would also
change which points the sampler visits.

**What a fix should consider**, in rough order of expected value:

1. **Scale the sample to the candidate's kernel count.** `n_want = min(64, max(16, trials_per_space))`
   is 40 here regardless of whether a variant compiles one kernel or four. Sampling fewer
   variants when each is expensive keeps the screen affordable.
2. **Screen only the shared-affecting subgrid.** Two variants differing only in `NUM_WARPS` or a
   cache hint produce the same `metadata.shared`; compiling both is waste. The sampler could
   project each candidate config onto its shared-affecting knobs and skip duplicates.
3. **Reuse across a space expansion.** ~~Worth checking whether the expansion changes the source
   text and thus misses every cache entry.~~ **Checked (2026-09-09), and the hypothesis is
   wrong**: the cache DOES hit across expansions. Ground truth from the worker job payloads on
   L3:21 (`jobs/*-prescreen-*.json`, counting `extra_kernel_src_paths`): every candidate's
   second prescreen sent 30–40 kernels instead of a fresh 40 — i.e. 0–10 cache hits, mean ~10%.
   The overlap is small not because the cache misses but because the expansion widens some
   domains' choices, and the same-seeded `rng.choice` sequence diverges from the first expanded
   knob onward, so the 40 sampled configurations are almost all new. Materialization only
   rewrites the PARAMS span, so identical values still produce identical source and hit. There
   is no recoverable half here; items 1 and 2 are the whole fix surface. (A cheap variant of 2
   that would raise the overlap: sample the second prescreen from the OLD configurations first
   and only top up with new ones — but that biases the screen away from exactly the region the
   expansion added, so it needs thought, not a quick patch.)

Item 3 is verified above: nothing to recover there, so a D5 fix is items 1 and 2 only.
