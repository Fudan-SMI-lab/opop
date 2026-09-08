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

**Evidence.** `RewriteCandidate.backend`, the `_detect_backend` override, and
`BACKEND_DECLARATION_MISMATCH` landed in `1ef142d`, after `run-l3-21-20260908-232211` started.
Driver-side code cannot reach a running orchestrator, so that run's rewrites still inherit the
parent backend.

**Why not now.** Nothing is broken; the feature simply has no evidence yet. Restarting a healthy
run to gather it would cost more than waiting for the next one.

**What to do.** Count `BACKEND_DECLARATION_MISMATCH` and CUDA-declared candidates on box 2's
L3:43 run and any later run. **Zero CUDA candidates is a legitimate result to report**, not a
failure — 35 of 35 Triton candidates in earlier runs was a consequence of what the prompts
asked for, and this is the first run in which the alternative is expressible at all.

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
