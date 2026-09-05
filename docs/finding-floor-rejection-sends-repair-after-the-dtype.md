# Finding: a floor-passing rejection sends repair to change the dtype, and it has never recovered

Observed live at 14:28–14:33 on `cand-90886b3c`, and it chains two separately-documented defects
into one candidate. This is the clearest instance in the record of how the noise-floor gate
actually costs candidates — not by rejecting them once, but by what the rejection makes repair do.

## The sequence

```
14:28:38  SPACE_REJECTED   attempt 0   dtype=fp16
            vs ieee ref  frac_within_tol 0.986913   max_abs_diff 2.822e-04
            vs tf32 ref                  0.980424                3.202e-04
            reference's OWN spread       0.976682                3.907e-04   <- the floor
14:28:39  repair dispatched (1 second later)
14:31:27  REPAIR_PRODUCED
            diagnosis: "The default fp16 compute path did not match either accepted reference
                        precision closely enough ... adding quantization absent from the
                        reference's fp32 tensors"
            change:    "Changed the default COMPUTE_DTYPE from fp16 to tf32"
14:33:37  SPACE_REJECTED   attempt 1   dtype=tf32
            OutOfResources: shared memory, Required: 115712, Hardware limit: 101376
14:33:38  repair dispatched again
```

Read the numbers against the diagnosis. The candidate agreed with the ieee reference on **98.69%**
of elements, with a *smaller* `max_abs_diff` than the reference's own two-precision spread, while
the reference agreed with itself on 97.67%. The repair agent's diagnosis — "did not match closely
enough" — is exactly what the failure message told it, and it is false. The kernel was more
consistent than the target it was being compared to.

Then the fix made it strictly worse. Switching `COMPUTE_DTYPE` fp16 → tf32 doubles every staged
element from 2 bytes to 4, and the candidate's space (`PROJ_*` and `ATTN_*`, two kernels) carries
**no shared-memory constraint at all** — the defect in
`finding-shared-memory-constraints-never-fire.md`. So the witness went from "numerically fine but
rejected" to "cannot compile", needing 115,712 bytes against a 101,376 limit.

Two defects, one candidate: the gate produced a false diagnosis, and the missing constraint let the
resulting fix be un-compilable.

## It is not a one-off — 3 for 3, and none recovered

Every candidate whose successive rejections show a dtype change:

```
l3-21  cand-6b313c39
  attempt 0  dtype=tf32  witness_default_failed   cand=0.958919 floor=0.955360  +0.0036
  attempt 1  dtype=fp16  witness_minimal_failed   cand=0.978946 floor=0.955360  +0.0236
  attempt 2  dtype=fp16  witness_minimal_failed   cand=0.978946 floor=0.955360  +0.0236
  attempt 3  dtype=fp16  witness_minimal_failed   cand=0.978946 floor=0.955360  +0.0236

l3-21  cand-89fa74fe
  attempt 0  dtype=tf32  witness_default_failed   cand=0.977751 floor=0.955360  +0.0224
  attempt 1  dtype=fp16  witness_minimal_failed   cand=0.978934 floor=0.955360  +0.0236
  attempt 2  dtype=fp16  witness_minimal_failed   cand=0.978934 floor=0.955360  +0.0236
  attempt 3  dtype=fp16  witness_minimal_failed   cand=0.978934 floor=0.955360  +0.0236

l3-43  cand-90886b3c
  attempt 0  dtype=fp16  witness_default_failed   cand=0.986913 floor=0.976682  +0.0102
  attempt 1  dtype=tf32  witness_default_failed   OutOfResources 115712 > 101376
```

**Every attempt is above the floor.** Every one. Ten rejections across three candidates, all of
kernels more consistent with the reference than the reference is with itself, and the repair loop
switched precision in all three cases — in opposite directions (tf32→fp16 twice, fp16→tf32 once),
because the failure message gives it no way to know which way is right.

And the outcome: `cand-6b313c39` and `cand-89fa74fe` exhausted all 4 attempts and were discarded.
Those are the two empty families that ended `run-l3-21-20260905-071312` about ten hours early
(`finding-run-stops-with-budget-unused.md`).

`cand-90886b3c`'s third attempt (14:35:42) then diagnosed the situation correctly, which is worth
recording because it is the loop working as designed once given a true signal:

> "The default witness fails before numerical comparison because the flash-attention kernel's
> two-stage pipeline requires 115,712 bytes of shared memory, exceeding the GPU's 101,376-byte
> limit. **The rejected precision repair exposed this independent launch-resource issue**; the
> reference conventions and train mode do not require any further computational change."
>
> Change: "Changed only the default `ATTN_PIPE_STAGES` from 2 to 1 to reduce attention-kernel
> shared-memory usage."

That is an accurate reading, including the observation that its own predecessor's fix created the
new problem. The repair agent is not the weak link — attempt 2 fixed a real resource error in one
try, exactly as it did 4 of 4 times in this run
(`finding-witness-has-no-resource-precheck.md`). What it cannot do is fix a correctness failure
that is not a correctness failure.

**Repair's recovery rate on floor-passing correctness rejections is 0 of 3.** Compare 4 of 4 on
genuine resource failures. The difference is not competence — a resource error names a true fact
("115712 > 101376") while a floor-passing correctness rejection names a false one, and the agent
correctly acts on whichever it is given.

Note the cost structure this creates: attempt 0 was a false signal, attempt 1's fix caused a real
failure, attempt 2 fixed *that*. Three agent calls and ~7 minutes, and the candidate is now back to
where it started numerically but with `ATTN_PIPE_STAGES` cut from 2 to 1 — a configuration change it
did not need, imposed by a chain that began with a rejection it should not have received.

## Attempt 2's numbers refine the finding — `frac_within_tol` alone is not enough

Attempt 2 (14:38:03) cleared the OOM and was rejected on correctness again, at
`frac_within_tol 0.978924` against the floor's 0.976682 — still above it, so by the single-metric
ledger this is a fourth floor-passing rejection. But comparing *all four* metrics against the
floor's own values tells a different story than attempt 0 did:

| metric | attempt 0 (fp16) | attempt 2 (tf32) | floor | |
|---|---|---|---|---|
| `frac_within_tol` | 0.9869 | 0.9789 | 0.9767 | both better |
| `median_rel_err` | 2.39e-04 | **1.34e-03** | 3.83e-04 | 0 better, **2 worse** |
| `max_abs_diff` | 2.82e-04 | **1.05e-03** | 3.91e-04 | 0 better, **2 worse** |
| `p99_rel_err` | 1.30e-02 | 2.05e-02 | 2.31e-02 | both better |

Attempt 0 beat the floor on **all four**. Attempt 2 beats it on two and is **2.7× worse on the
typical element** (`median_rel_err` 1.34e-03 vs 3.83e-04) and 2.7× worse on the largest deviation.
So attempt 2 is genuinely less accurate than the reference-vs-itself even though its
`frac_within_tol` sits above the floor — the repair chain did degrade the kernel, exactly as
`opop-v2-noise-floor-gate-damages-candidates` predicts, and the single metric I had been ledgering
does not show it.

**That is a limitation of my own analysis, not just of the gate.**
`measurement-rejections-above-the-noise-floor.md` classifies rejections by `frac_within_tol`
alone, which is the metric the gate uses — so as a description of *what the gate did* it is right.
But as evidence that a rejected candidate was *fine*, one metric is not enough: a candidate can
clear the floor on `frac_within_tol` while being multiples worse on median and max deviation.

So I ran the four-metric comparison over the whole record rather than leaving that as a caveat:

| | n |
|---|---|
| **clean on ALL four metrics** (better than the reference-vs-itself everywhere) | **10** |
| above the floor on `frac_within_tol` but worse on ≥1 other metric | **1** |
| below the floor on `frac_within_tol` | 18 |

```
l3-21  cand-6b313c39   attempts 0,1,2,3   4/4  CLEAN
l3-21  cand-89fa74fe   attempts 0,1,2,3   4/4  CLEAN
l3-43  cand-90886b3c   attempt  0         4/4  CLEAN
l3-48  cand-61f768c8   attempt  2         4/4  CLEAN
l3-43  cand-90886b3c   attempt  2         2/4  gate-only  <- the degraded one
```

The single ambiguous case is exactly the one that prompted the check — attempt 2, the product of two
repairs. Every *other* above-floor rejection is clean on all four metrics, and both the 18
below-floor rejections are 0/4 (none of them beats the floor on any metric), so the two groups
separate cleanly rather than shading into each other.

My "1 verified clean, 9 unverified" reading a paragraph ago was too pessimistic: it is **10 verified
clean, 1 genuinely degraded**. Worth stating plainly because I nearly recorded a weaker conclusion
than the data supports — the four-metric check made the ledger *stronger*, not weaker, while
correctly flagging the one row that deserved it.

This does not change the deferred decision, but it does sharpen what a floor-relative gate would
have to compare. A rule keyed on `frac_within_tol` alone would have accepted attempt 2, which is
2.7× worse on the typical element. A rule requiring the candidate to match the reference-vs-itself
on **median and max deviation as well** accepts all 10 clean cases and correctly rejects attempt 2 —
and that is a materially different (and better-founded) proposal than the single-metric one in
`decisions-awaiting-user.md` item 1.

## Why the dtype switch is the specific damage

`opop-v2-noise-floor-gate-damages-candidates` already records that repair can break a correct
kernel (L3:48, 0.976 → 0.844 + NaN). This adds the *mechanism*, and it is narrower than
"repair makes mistakes":

A correctness failure with no reproducible logic bug leaves the agent one lever it can always
pull — precision. It is the knob the prompt tells it to expose (`modules.py:334`), the one most
plausibly connected to a numeric mismatch, and the change is a one-line edit. So a false
correctness signal reliably produces a precision change, and a precision change on this hardware
is never free:

- **fp16/bf16 → tf32/ieee** doubles staged bytes (this case: instant OOM) and abandons the
  tensor-core path that `opop-v2-fp16-knob-gap` and this run's 11.0 ms result both depend on.
- **tf32 → fp16** halves the bytes but *increases* the numeric deviation, which is the thing the
  agent was told to fix — so it fails again for the same stated reason, which is exactly the
  `cand-6b313c39` / `cand-89fa74fe` pattern: switch to fp16, then fail the minimal witness three
  times running at an unchanged 0.978946.

That last detail is worth its own line. Both L3:21 candidates report the **identical** frac to six
decimal places across attempts 1, 2, and 3 — `0.978946`, `0.978946`, `0.978946`. Three repair
attempts, three agent calls, and the measured deviation did not move at all. The loop was not
converging on anything; it was re-measuring the same corner.

## What I am not doing, and the one thing that might be separable

**Not touching the gate.** Third accumulation of evidence, user declined twice
("问题4现阶段不要实施, 危险性过大"), and `measurement-rejections-above-the-noise-floor.md` shows why
the tolerance question is genuinely hard: 13 of 18 below-floor rejections sit within 0.0021 of the
floor, and L3:21's best kernel ever is one of them.

**Not touching repair's prompt either.** The tempting fix is "tell repair not to change the dtype",
but that is wrong: `finding-k-expansion-drops-constraints.md` records `STATS_BLOCK_M=256` being
infeasible at tf32 and holding the best result at fp16, so precision *is* sometimes the right
lever. Forbidding it would break the cases where it works.

The separable thing, flagged and not done: **the failure message already contains the floor
comparison** — the detail string prints "reference's OWN ieee-vs-tf32 spread (task noise floor, NOT
a bug)" right there. So the harness knows, at the moment it dispatches repair, that this candidate
is above the floor. Suppressing the repair dispatch in that case would:

- cost nothing in accept/reject semantics — the candidate is still rejected, it just is not
  "fixed";
- save 3 agent calls and ~5 min per occurrence (10 rejections in the record → ~50 min);
- stop producing the un-compilable tf32 variants;
- and leave the actual decision (should it have been accepted?) entirely to the user.

That is ~15 lines in the orchestrator's repair branch, gated on a comparison the event already
carries. It is the smallest change that acts on this evidence without pre-empting the deferred
decision, and I am putting it on the pending list rather than making it — it still turns on
believing the floor comparison, which is the same premise the user deferred.

`scripts/audit_noise_floor_rejections.py` prints the ledger; the dtype-switch sequences above come
from grouping `SPACE_REJECTED` details by candidate.
