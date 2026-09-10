# A800 per-task noise floors: measured, and they match box 1 almost exactly

**Measured** 2026-09-10 on box 3 (A800 80GB PCIe, sm_80) at commit `8eb8d16`, via
`scripts/probe_noise_floor.py` (5 trials/task). Raw readings:
`/root/autodl-tmp/work/opop-glm/runs-l3/noise_floors_a800.json` on that box.

**Why this was measured**: `docs/box-a800-setup.md §8` listed the three ieee-vs-tf32 floors
(0.9554 / 0.9767 / 0.9778) as **box-1 numbers** that "must be re-measured here before use", on the
reasoning that a floor is a *(card, task)* property. J2-5 and J1-4 both read "final_reeval_ms new ≤
old × (1 + noise floor)", so an imported floor would silently set the acceptance width of both
control runs.

---

## 1. The readings

| task | A800 min | A800 max | box 1 | Δ vs box 1 | control | verdict |
|---|---|---|---|---|---|---|
| level3:21 | **0.955298** | 0.955369 | 0.9554 | **−0.0001** | 1.000000 | measured |
| level3:43 | **0.976460** | 0.977026 | 0.9767 | **−0.0002** | 1.000000 | measured |
| level3:48 | **0.979819** | 0.979859 | 0.9778 | **+0.0020** | 1.000000 | measured |

Per-trial spread is tiny — L3:21 ranges 0.955298–0.955369 over five trials, i.e. **7e-5** — so these
are not five draws from a wide distribution that happened to land together.

## 2. The finding: the floor did NOT move between cards

Two of the three agree with box 1 to **within 0.02%**, the third to **0.2%**. Against the *reason*
for re-measuring, that is a negative result, and it is worth stating plainly:

**On these three tasks, across sm_89 → sm_80, the ieee-vs-tf32 floor behaves as a property of the
TASK rather than of the (card, task) pair.**

That is not what I expected. The A800's tensor-core ratio is very different (tf32/fp32 = 5.9× here
versus 1.6× on the 4090), its L2 is smaller (40 vs 72 MiB) and its Triton is a different major
version. None of that moved the floor.

**Why it is nonetheless coherent**: the floor measures how far the *reference's own* output moves
when the fp32 matmul path switches between tf32 and ieee. That is dominated by **tf32's mantissa
truncation** — 10 explicit bits, which is an arithmetic property of the tf32 format, not of the card
— and by the reference's own numerical structure (how many accumulations, over what magnitudes). The
card decides how FAST that arithmetic runs, which is why the ceilings differ by 2–6× while the floor
does not.

**What this does NOT license.** Two boxes and three tasks is not a rule, and the mechanism above
predicts its own exceptions: a card whose tf32 mantissa or accumulation width differed, a task whose
reference is close to a cancellation, or a torch version that changed a reduction order, would all
break it. The probe costs about a minute per task and needs no agent calls. **Keep measuring it on a
new box** — the cheap thing here is the measurement, not the assumption.

## 3. What was ruled out, rather than assumed

A floor number is only meaningful if two specific things are true, and both were checked in the same
run rather than argued:

**RNG inside `forward()`.** If the reference draws random numbers in its forward pass, the two
precision calls see different noise and the "precision floor" is really RNG noise. The probe
re-seeds immediately before **each** forward, and separately runs the reference twice at the **same**
precision as a control. **All three controls read exactly 1.000000**, so there is no
nondeterminism to confound the reading. This is a live hazard rather than a hypothetical: the
harness's own witness loop seeds once and then calls the reference twice, so a task with active
dropout would be compared against a moving target there.

**The precision switch not taking effect.** An old torch where the API moved, or a task with no
matmul/conv at all, would give floor = 1.0 — which reads as "this task is precision-insensitive"
when it may mean "nothing was measured". All three floors are well below 1.0 **and** the controls are
at 1.0, so the switch demonstrably acts. Had both been 1.0 the probe reports `INCONCLUSIVE`, not a
floor.

## 4. The operational consequence is unchanged, and it is the important part

All three floors sit **below** `evaluation.relaxed_pass_frac = 0.99`. So on this box, exactly as on
box 1:

**The absolute relaxed gate is unreachable for a low-precision candidate on all three L3 tasks,
however correct that candidate is.** Acceptance depends on the fp64 relative arm
(`fp64_relative_gate: true`), which is therefore load-bearing here and not a safety net.

That arm has been measured to hold its weight rather than merely to admit more: rescued candidates
all had ratios < 1.0 (i.e. **closer to fp64 truth than the reference itself**), while adversarial
defects were rejected by factors of 28–290×.

## 5. Where these numbers are now used

- `configs/experiments_l3_glm_a800.yaml` — the comment that flagged the floors as unmeasured box-1
  values is updated to point here.
- `docs/box-a800-setup.md §8` — the open item is closed.
- The width for **J1-4 / J2-5** ("final_reeval_ms new ≤ old × (1 + floor)") on this box is
  **0.9553 / 0.9765 / 0.9798** for L3:21 / 43 / 48.

## 6. What this did not verify

- **Only the three L3 tasks.** L1 and L2 floors on this card are unmeasured.
- **Only the ieee-vs-tf32 pair.** The fp16/bf16 floors — the precisions candidates actually win with
  here — are a different measurement and were not taken.
- **n = 5 per task.** Enough to show the spread is ~1e-4, not enough to characterise a tail.
- **Nothing about `fp64_rel_multiplier`.** Whether 2.0/3.0 are the right multipliers **on this card**
  is untouched by this probe.
