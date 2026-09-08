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

## D2. `STORE_DTYPE=fp16` is a genuine defect the space still offers

**Evidence.** L3:21, per-combination outcomes over 40 trials:

| compute / store | complete | failed |
|---|---|---|
| fp16 / fp16 | 7 | 0 |
| fp16 / fp32 | 2 | 0 |
| tf32 / fp16 | 12 | 1 |
| tf32 / fp32 | 4 | 0 |
| ieee / fp16 | 3 | 2 |
| ieee / fp32 | 1 | 0 |
| **bf16 / fp16** | **0** | **5** |
| **bf16 / fp32** | **0** | **3** |

The failure detail shows two different magnitudes. `bf16/fp32` fails at
`ratio_to_reference: 8.000` against a 3.0 multiplier — a real but modest error. Several
`*/fp16` store combinations fail at `ratio_to_reference: 1184` to `1237` — three orders of
magnitude out, with `frac_within_tol: 0.007` and `cosine: 0.653`. That is not a tolerance
question; storing an intermediate in fp16 destroys the result for this task.

**The gate is behaving correctly here** — this is worth stating explicitly, because the
opposite (a correct kernel rejected by an over-tight gate) has happened before on other tasks
and is recorded in memory. Both the absolute arm and the fp64-relative arm reject these, the
reference's own ieee-vs-tf32 noise floor is reported alongside (`frac 0.955`, `cosine
0.99999975`), and the failing candidates are 8x to 1237x worse than the reference's own
distance from an fp64 golden. Nothing here needs loosening.

**Why not now.** The candidate offering a fatal store dtype is a candidate-quality issue, and
the harness handles it correctly by rejecting those trials. The cost is the same budget waste
as D1 and is counted there.

**What a fix might do.** This is really the same lever as D1 — a store dtype that never once
passes is a hopeless categorical value. It is listed separately because it also suggests a
*prompt* change: the contract tells candidates to keep the accumulator in fp32 but says nothing
about intermediates written to global memory between kernels, which is what `STORE_DTYPE`
controls here. A sentence naming that distinction is low risk, but it is a prompt edit that
would reach a running experiment mid-flight (the `.md` files are re-read per call), so it waits
until no run is in progress.

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
