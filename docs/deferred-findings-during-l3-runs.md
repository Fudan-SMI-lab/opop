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

## D6. A measured zero was recorded as "not measured" — FIXED (`03ab1a3`)

Recorded here because it is the answer to the deferred item *"investigate why `n_spills` is
unmeasured on some candidates"*, and the answer turned out to be a defect rather than a collector
gap. `if n_spills:` in `bottleneck.py` dropped a measured **zero**, so 11 of box 2's 128
`DIMENSION_STATE` records said `"spill count not read from the compiler"` about candidates whose own
best trial carried `n_spills: 0`. It fires on exactly the HEALTHY candidates, which is why it looked
occasional. Full account in the commit; 6 tests and 3 revert variants.

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

**PARTIALLY ADDRESSED (`59d5a71`).** That commit fixes a DIFFERENT half of the same area — the
runaway case, where a batch borrowed `build_timeout_s` (1200 s) and answered nothing at all (box 3:
two batches at 1200.5 s and 1201.0 s, 0.67 h for zero verdicts). The screen now has its own
`base + per_config * n` deadline. **The pricing problem above is untouched**: items 1 and 2 remain open,
and they are the ones that would make the screen net POSITIVE rather than merely bounded. See
`docs/plan-after-the-paired-runs.md` §B2 for why they wait for a gap between experiments.

---

## D7. The prescreen cap is now measured in production — bounded, but sized close to the wire

**Evidence, from the three E1/E3 runs in flight (2026-09-12, all at `842e2a6`).** The
`59d5a71` cap is doing exactly what it was built for, and the safety argument holds on disk:

| box | run | prescreens | answered | elapsed | timed out | configs excluded by a timed-out screen |
|---|---|---|---|---|---|---|
| 1 | run-l3-43-20260911-230217 | 2 | 40, 5 | 130.4 s, 135.4 s | 0 | — |
| 2 | run-l3-43-20260911-230736 | 2 | 40, 0 | 148.9 s, 150.3 s | 1 | **0** |
| 3 | run-l3-48-20260911-231217 | 1 | 0 | 150.2 s | 1 | **0** |

**The fix works.** The runaway tail is gone: box 3's earlier run spent 1200.5 s and 1201.0 s on
batches that answered nothing (0.67 h for zero verdicts); the same situation now costs 150 s. And
the property the design rests on is confirmed rather than merely argued — **the two timed-out
screens excluded 0 configurations**, because a timeout caches nothing, so every unanswered
configuration still received a real trial with the full `build_timeout_s`.

**What is worth recording.** Two batches that DID answer finished at 130.4 s and 148.9 s against a
150.0 s cap — the second with 0.7% of margin. That is not a defect and it did not cost anything
(both answered in full, 16 and 12 infeasible configs found), but it says the budget
`30 + 3 * n` is sized close to the real cost of a 40-config batch on these candidates, not
comfortably above it. A slightly heavier candidate would time out and simply fall back to
"every config gets a trial", which is the safe direction — it loses the screen's savings, not any
part of the search space.

**Why not now.** Raising `prescreen_per_config_timeout_s` would change what gets screened, and
therefore which trials run: a run started after it is not comparable with one before, and E1's two
arms must remain comparable with each other. It also interacts directly with D5's untouched
**pricing** problem (items 1 and 2), so the two should be decided together in the gap between
experiments rather than separately.

**What a fix would have to do.** Not simply raise the constant. The per-config term should scale
with the candidate's kernel count — a multi-kernel variant compiles several kernels per config,
which is the same root cause as D5 item 1 — so that a heavier candidate gets a proportionally
larger budget instead of one flat number that is generous for a single-kernel candidate and tight
for a four-kernel one.

---

## D8. One candidate's pathological PTX ate 43% of a run's wall clock — bounded, but very expensive

**Evidence, box 1's E1 control arm (`run-l3-43-20260911-230217`), read live at 04:55 on 2026-09-12,
5.16 h into a 12 h budget.** Wall time between consecutive `TRIAL_DONE` events, attributed to the
candidate whose trial it was:

| candidate | trials | wall time | share | complete/fail | per trial |
|---|---|---|---|---|---|
| **cand-941ea454** | **12** | **1.68 h** | **43.2%** | 9 / 3 | **8.4 min** |
| cand-e254236c | 80 | 0.96 h | 24.7% | 68 / 12 | 0.7 min |
| cand-772ea591 | 80 | 0.85 h | 21.9% | 57 / 23 | 0.6 min |
| cand-fdfbcb59 | 40 | 0.40 h | 10.3% | 27 / 13 | 0.6 min |

**14x the per-trial cost of its three siblings, for 12 trials against their 80/80/40** — and it is
also the arm's WORST candidate (best 6.765 ms against the leader's 4.244 ms). The three slowest gaps
in the whole run are all its: 32.9 min, 50.1 min, and a third compile still running at 24 min when
this was written.

**Root cause, observed directly rather than inferred.** The live worker's child was

```
ptxas -lineinfo -v --regAllocOptLevel 2 --gpu-name sm_89 /tmp/tmpzap322op.ptx
    RSS 12.2 GB, 23:43 of CPU in 23:43 elapsed (100% busy, not blocked)
    input: 74474 lines / 3.56 MB of PTX
```

Same class as the recorded `agent-script-can-oom-the-whole-box` case (272341 lines took ptxas to
111 GiB), one order of magnitude smaller: this box has 755 GB so 12.2 GB was never dangerous, and
`free` showed 309 GB still free. **Nothing is wedged and nothing is at risk** — `ptxas` is simply
spending tens of minutes on register allocation for a kernel whose PTX is enormous.

**The existing guard IS working.** The 1800 s job deadline caught the worst one:
`job cand-941ea454-tr-ac5c798e-eval-aa4e2f3f exceeded 1800.0s` → recorded as `failure_kind:
timeout`. So the cost is bounded per trial; it is not an unbounded hang.

**Why not now.** This costs budget, not correctness: the trials that complete are correctly
measured, the leader is chosen from them, and both E1 arms are subject to the same mechanism. Any
fix changes which configurations get evaluated (or how long they may take), so a run started after
it is not comparable with one before — and E1's two arms must stay comparable with each other. The
`prescreen` cannot help here either: it is compile-only and would pay the SAME ptxas cost.

**What a fix would have to do, and what it must not do.** It must not narrow the search space or
lower `build_timeout_s` — a legitimate candidate whose ptxas genuinely needs ten minutes must still
be allowed to finish (`never-narrow-the-search-space-to-control-cost`, and the explicit warning in
`prescreen_timeout_s`). The generalizable options, in order of how well the evidence supports them:

1. **Report per-candidate cost so the operator and the report can see it.** Currently nothing in
   the event log states that one candidate consumed 43% of a run; it took a bespoke script to find.
   Purely additive, changes no decision, and is the prerequisite for judging any of the below.
2. **Cap PTX size before invoking ptxas**, per kernel, as a *screen* whose failure is a normal
   trial failure with a stated reason — 74 k lines against a corpus median in the hundreds is a
   detectable outlier. Needs the distribution measured first; a threshold guessed here would be the
   same mistake as the 10x constant.
3. **Let a candidate's measured per-trial cost feed its trial ALLOCATION** rather than its
   admission — i.e. a candidate at 14x the cost gets fewer trials, not zero. This one interacts
   with `budget-is-per-space-not-per-candidate` and needs its own design.

### D8 update, read live at 05:40 — it worsened, and the decision was to let it run

Re-measured 1.5 h later, same run:

| | at 04:55 | at 05:40 |
|---|---|---|
| share of trial wall time | 43.2% | **53.4%** |
| per trial | 8.4 min | **10.8 min** |
| trials done | 12 | 14 |
| ptxas peak RSS | 12.2 GB | **37.6 GB** |
| timeouts recorded | 1 | 2 |

**Not an OOM risk**: 37.6 GB against 667 GB available on a 755 GB box (5.6%), no swap, no OOM kill
in the box's history. The 1800 s deadline keeps cutting the worst compiles off.

**The contrast that makes this a per-candidate pathology rather than a task property.** Box 2's
treatment arm, same task, same 12 h, at the same point in the run:

| arm | candidates | per-trial cost | rewrite rounds |
|---|---|---|---|
| box 1 control | 4 | 0.6 / 0.6 / 0.7 / **10.8** min | **0** |
| box 2 treatment | 8 | 0.6–1.1 min, all of them | 1 |

**The budget consequence, projected from the measured rate.** At 6.64 h of 12 h used,
`cand-941ea454` had 14 of its 40 trials done. The remaining 26 at 10.8 min is **4.7 h** against
**5.4 h** of budget left — it would consume nearly all of it, and the control arm has recorded zero
rewrite rounds so far.

**DECISION (user, 2026-09-12): do not intervene; let both arms run their full 12 h.** The reasoning
is that arm parity is worth more than this one arm's rewrite rounds: killing the candidate, or the
run, would make the two arms structurally different and cost the comparison entirely, whereas
letting it finish leaves every other measurement from the pair clean and usable (the conversion
rates, the A1 threshold in production, this evidence itself). **The accepted risk is explicit: E1's
control arm may end with 0 rewrite rounds, in which case S2d(c) has no control data and that pair
must be re-run after D8 is fixed.** That is a known cost, not a surprise.

**05:53 — the projection tightened from "may" to "near-certain".** The third 1800 s timeout fired on
the same candidate and the arm recovered from it, so the mechanism is behaving exactly as designed;
what changed is the arithmetic. At 6.89 h of 12 h:

    share of trial wall time   60.4%   (43.2% -> 53.4% -> 60.4% over three readings)
    per trial                  13.5 min
    trials done                15 of 40
    remaining 25 trials        5.6 h needed against 5.11 h of budget left

So **the wall clock, not the trial budget, will end this candidate**, and the control arm will reach
its 12 h having never entered a rewrite round. The re-run of this pair after a D8 fix should
therefore be planned for rather than treated as contingent — the decision to protect arm parity
stands, but its cost is now a fact rather than a risk.

One thing this does buy: the arm is an unusually clean natural experiment for D8 itself. Three
sibling candidates at 0.6–0.7 min/trial completed 40, 80 and 80 trials in the same run where one
candidate at 13.5 min completed 15, all under identical settings — which is the controlled
comparison any per-candidate cost mechanism would need as its evidence base.

### D8 correction, 07:50 — the mean was the wrong statistic and it changed what a fix must target

Every reading above reported this candidate's **mean** per-trial cost, which rose 8.4 → 10.8 →
13.5 min and read as "this candidate is uniformly 14x slower than its siblings". Pulling the
per-trial breakdown instead of the aggregate falsifies that framing:

```
cand-941ea454, trial-by-trial wall cost (min)
  02:42  10.6  complete      03:19   0.8  complete      05:02  50.1  fail timeout
  02:42   0.0  complete      03:19   0.6  complete      05:53  50.1  fail timeout
  02:42   0.5  fail infeasible_shared_memory            06:43  50.1  fail timeout
  02:43   0.8  complete      04:09  50.1  fail timeout   06:44-06:49  five trials at 0.8-1.8
  02:45   2.2  complete      04:10   0.7  complete
  03:18  32.9  fail runtime_error
                                        median 1.1 min      mean 12.3 min
```

**The median is 1.1 min — within the 0.6–0.7 min band of its three siblings.** 16 of 21 trials are
normal; the entire excess sits in **5 catastrophic trials** (32.9 + 50.1 × 4 = **233 min**, 94% of
the candidate's total cost). So this is not "a slow candidate", it is **a normal candidate that
occasionally emits pathological PTX**.

This matters because it invalidates D8's option 3 above. A per-candidate cost mechanism that
allocates trials by *measured average* cost would have throttled a candidate whose typical trial is
as cheap as everyone else's — punishing 16 good trials for 5 bad ones, and doing it on a statistic
(the mean) that a handful of outliers control. **The unit of the pathology is the trial, not the
candidate**, so a fix has to act per compile, which is what options 1 and 2 already do. Option 3 is
withdrawn unless it is re-specified on a robust statistic *and* on the per-trial unit.

Same lesson as `tuning-objective-must-be-median` in a new place: an aggregate over a
heavy-tailed distribution names the tail, not the thing.

### D8 second correction, 08:00 — the budget projection was wrong too, and in the same way

The 05:53 reading projected "25 remaining trials need 5.6 h against 5.11 h left ⇒ the control arm
reaches 12 h with 0 rewrite rounds", and called that "near-certain". **It did not happen.** Read at
08:00, 8.90 h in:

| | projected at 05:53 | actual at 08:00 |
|---|---|---|
| `cand-941ea454` trials | 15 of 40, ~25 left needing 5.6 h | **37 of 40** |
| the last 10 trials | — | **10 consecutive completes in 8 min** |
| remaining need | 5.6 h vs 5.11 h left | **0.45 h vs 3.10 h left** |
| rewrite rounds reachable | no | **yes, ~2.65 h of headroom** |

**Why the projection failed.** It multiplied the remaining trial count by the *mean* per-trial cost
— the same statistic the correction above shows is controlled by five outliers. Once those five had
fired, the candidate reverted to its median 1.1 min and burned through 22 trials in the time the
projection said would buy four. Measured over all 72 of its jobs the hang rate is **12/72 = 16.7%**,
so the honest projection was never "13.5 min × N" but "N × (2 jobs × (83% × 25 s + 17% × 1500 s))"
= 9.0 min/trial — which is what actually happened.

**Consequence for the plan.** The re-run of the E1 pair, which the 05:53 entry said "should be
planned for rather than treated as contingent", is **contingent again**: if the control arm enters
even one rewrite round, S2d(c) has control data and the pair may be usable as-is. Box 2's treatment
arm already has three family rounds with reconciliation (7/15 → 13/16 → 11/15 hits) and gains of
11.4% → 20.8% → 31.7%. Whether the pair is usable is decided by what box 1 does in its last 3.1 h,
not by this projection — so the decision waits for the run to end.

The user's "do not intervene" decision was right for a reason neither of us had at the time: the
candidate was never uniformly expensive, so waiting cost far less than the mean implied.

---

## D9. The single-config compile screen borrows a real trial's 1200 s budget — 1200 s + 1800 s = the 50 min gaps

**This is the same defect `59d5a71` fixed in `prescreen_batch`, in the sibling function that commit
did not touch.** `compile_screen` (`correctness.py:268`) passes `self.cfg.build_timeout_s` to
`run_job`, while `prescreen_batch` (line 180) passes `prescreen_timeout_s(...)` — a purpose-built
`30 + 3n` budget clamped by `build_timeout_s`. The docstring on `prescreen_timeout_s` argues at
length why a screen must not borrow a trial's compile budget; that argument applies verbatim to
`compile_screen`, which was never brought under it.

**How it was found.** Chasing box 1's four 50.1-minute gaps between `TRIAL_DONE` events. The gaps
contain no intervening event, and `SPACE_PRESCREENED` all fired before 02:45, so a batch prescreen
could not explain them. The job directory does, exactly:

```
04:12:07  cand-941ea454-compile-screen-fe8e01e7.json      <- written, NO .out.json ever
04:32:08  cand-941ea454-tr-7038a5a4-eval-758c300e.json    <- 04:12:07 + 1200 s = 04:32:07
05:02:11  cand-941ea454-compile-screen-781a63fa.json      <- 04:32:08 + 1800 s = 05:02:08
```

**1200 s (screen deadline) + 1800 s (eval deadline) = 3000 s = the 50.1 min.** Every one of the five
hung screens is followed by a timed-out eval, 5/5.

**The screen's deadline is 11x its own p99.** Measured across all three in-flight runs
(1108 single-config screens, three tasks, two GPU models):

| box | screens | answered | hung | p50 | p90 | p99 | max | cost of the hung ones |
|---|---|---|---|---|---|---|---|---|
| 1 (L3:43, 4090) | 214 | 208 | **6** | 11.8 s | 28.5 s | 105.0 s | 991.0 s | **2.00 h** |
| 2 (L3:43, 4090) | 526 | 526 | 0 | 11.4 s | 26.8 s | 56.9 s | 167.2 s | 0 |
| 3 (L3:48, A800) | 368 | 367 | **1** | 13.6 s | 26.5 s | 38.7 s | 144.7 s | 0.33 h |

A **120 s** deadline would still answer 206/208, 524/526 and 366/367 — 99.0%, 99.6% and 99.7% of the
screens that answered at all — while capping the hung tail at a tenth of its current cost. And the
non-answer is free by construction: `compile_screen` returns `None` on anything but the compiler's
own figure exceeding the device limit, so a timeout removes nothing from the search (the same
three-valued property `cached_shared_verdict` relies on).

**The screen does not even warm the cache for the eval that follows it.** The one slow-but-answered
screen took **991.0 s** and its eval then took **978.8 s** — the compile was paid twice, in full.
Against a baseline of 146 pairs where the screen finished in ≤30 s and the eval's median was 13.8 s.
So there is no hidden benefit being bought by the long deadline.

**Scale of the waste, independent of the hangs.** Screens are a serial lane (1–2 of 419/976/833 job
windows overlap the next job's start, i.e. effectively none), so their wall time is additive:

| box | screen wall | share of run span | eval wall | share |
|---|---|---|---|---|
| 1 | 179.3 min | **34.1%** | 219.0 min | 41.6% |
| 2 | 138.7 min | **26.6%** | 156.4 min | 30.0% |
| 3 | 120.8 min | **23.4%** | 146.1 min | 28.3% |

**A quarter to a third of every run is spent compiling twice** — once for the screen, once for the
eval. The hung tail is 2.33 h of that; the rest is the duplicated compile itself. D5 measured this
same double-payment for the batch prescreen and found it net **−9.6 min** on L3:21; this is the
per-config version of the same accounting, an order of magnitude larger because it runs per trial.

**Why not now.** Changing the screen's deadline changes which configurations get a screen verdict
versus falling through to a real trial, and therefore the order and cost of what the sampler visits
⇒ **a run started after it is not comparable with one before**, and E1's two arms must stay
comparable with each other. This is the same reason D5 and B2 are deferred. Both E1 arms are subject
to the mechanism, but not equally — box 1 lost 2.00 h to it and box 2 lost 0 — which is itself worth
recording as a **confound in the E1 pair** (see below).

**What a fix must do.**
1. Route `compile_screen` through `prescreen_timeout_s(cfg, 1)` — one line, and it inherits the
   existing argument, tests and config fields rather than adding a fourth timeout constant. At
   n=1 that is 33 s, which is below the measured p90 of 26.8 s only just; the honest form is to let
   the *screen* deadline be `max(prescreen_timeout_s(cfg, n), floor)` with the floor derived from
   the measured distribution above (120 s covers 99.0–99.7%), not guessed.
2. It must **not** touch `build_timeout_s`. A legitimate candidate whose ptxas genuinely needs ten
   minutes must still get its full compile budget in the real trial
   (`never-narrow-the-search-space-to-control-cost`).
3. The deeper fix is to stop paying the compile twice — have the screen's compile artifact serve the
   eval, or drop the per-config screen in favour of the batch one. That is B2's territory and needs
   its own design; the deadline fix is the cheap, low-risk half.

**Confound recorded for the E1 pair.** The two arms did not pay this equally: 6 hung screens and
2.00 h on box 1, 0 and 0 h on box 2. That is a real asymmetry in effective budget between the arms,
caused by one candidate's PTX (D8) meeting this deadline (D9). It does not invalidate the pair —
both arms ran the same code with the same settings, and the asymmetry is an outcome of what the
generator produced, not of the treatment — but any claim about the arms' *search volume* must state
it, alongside `equal-configs-do-not-imply-equal-search`.

### D9 decomposition, 08:10 — the arm asymmetry is the timeouts, not the candidates

Measured on both arms at the same point (8.80 h and 8.85 h of span), separating *answered* job time
from *hung* job time:

| | box 1 control | box 2 treatment | ratio |
|---|---|---|---|
| complete trials | 180 | 422 | **2.34x** |
| complete trials / h | 20.45 | 47.58 | **2.33x** |
| answered jobs | 411, median **13.6 s** | 1017, median **12.9 s** | **1.05x** |
| total answered time | 2.60 h | 4.77 h | — |
| **wall lost to hung jobs** | **4.75 h** | **1.00 h** | **4.75x** |
| complete trials / h **excluding hang time** | 44.40 | 53.73 | **1.21x** |

**The 2.33x throughput gap collapses to 1.21x once hang time is removed, and the two arms' median
answered job differs by 5%.** So the arms are not running different-cost work — they are running
almost identically-priced work, and one of them spent 4.75 h of a 12 h budget on jobs that produced
no output at all. That is 39.6% of box 1's entire budget.

This changes how the confound must be described. It is not "box 1 happened to draw an expensive
candidate" — box 1's *answered* jobs are as cheap as box 2's. It is "**box 1 paid D9's deadline six
times and box 2 paid it zero times**", which makes the asymmetry an artifact of a harness defect
rather than a property of either arm's search. It also raises D9 from a budget-efficiency finding to
the **largest single measured waste in the run**: 4.75 h against the 2.33 h of duplicated-compile
overhead the screen costs even when it answers.

The 1.21x residual is the real, attributable difference between the arms (different candidates,
different spaces), and that is small enough that the pair remains usable if box 1 reaches a rewrite
round.

### D9 scope check — the screen's VERDICTS are sound; only its deadline is wrong

Worth stating explicitly, because it bounds what a fix may touch. Across both E1 arms, 225
`CONFIG_SCREENED_INFEASIBLE` refusals:

| | box 1 | box 2 |
|---|---|---|
| refusals | 51 | 174 |
| required/limit ratio, min | 1.07 | 1.05 |
| ratio, median | 1.62 | 1.62 |
| ratio, max | 4.53 | 6.48 |
| refusals within 5% of the limit (marginal calls) | **0 / 51** | **0 / 174** |
| trials that reached a launch and *still* hit shared-memory exhaustion | **0** | **0** |

**Not one refusal is marginal** — the closest asks for 5% more shared memory than the device has, and
the median asks for 62% more. And there are **zero false negatives**: no configuration the screen
passed then failed at launch for shared memory. So the screen is doing exactly the job D5 designed it
for, on the criterion it was designed for (the compiler's own `metadata.shared` against the device
limit), with no evidence of over- or under-rejection.

That isolates the defect precisely: **the decision logic is right and the deadline is wrong.** A fix
must therefore change only the timeout, and must not touch the refusal criterion, the caching rule,
or which configurations get screened. It also means the 4.75 h box 1 lost bought nothing at all —
those six screens were not protecting a hard call, they were compiling PTX that `ptxas` could not
finish inside 1200 s, for a verdict the real trial would then re-derive from scratch.


