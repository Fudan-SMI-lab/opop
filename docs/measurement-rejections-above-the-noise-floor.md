# Measurement: 10 of 28 correctness rejections are of candidates MORE consistent than the reference

New evidence on the deferred gate decision (`docs/decisions-awaiting-user.md` item 1; the user
declined twice — "问题4现阶段不要实施, 危险性过大"). Recorded because a live rejection at 14:28:38
made the asymmetry measurable in a way the earlier per-run cases did not, and because the numbers
cut **both** ways — they sharpen the case for the change and they identify exactly what the change
would cost.

## The live case

`cand-90886b3c`, `witness_default_failed`, 14:28:38:

```
vs ieee ref:   frac_within_tol 0.986913   cosine 0.99999998   max_abs_diff 2.822e-04
vs tf32 ref:   frac_within_tol 0.980424   cosine 0.99999995   max_abs_diff 3.202e-04
reference's OWN ieee-vs-tf32 spread (task noise floor):
               frac_within_tol 0.976682   cosine 0.99999993   max_abs_diff 3.907e-04
```

The candidate agrees with the ieee reference on **98.69%** of elements. The reference computed two
ways agrees with *itself* on **97.67%**. The candidate is more consistent with the reference than
the reference is with itself — on every one of the three metrics, including a smaller
`max_abs_diff` — and it was rejected for insufficient correctness.

A repair agent was dispatched one second later (14:28:39) to fix a kernel that is not broken. That
is the failure mode already recorded in
`opop-v2-noise-floor-gate-damages-candidates` (L3:48, 0.976 → 0.844 + NaN after repair).

## The count, across every L3 run

Every space or expansion rejection whose detail carries both the candidate's `frac_within_tol` and
the task's floor, classified by which is larger:

| | n |
|---|---|
| candidate **above** the floor — more consistent than the reference-vs-itself | **10** |
| candidate below the floor — genuinely less consistent | 18 |

So **36% of correctness rejections are of candidates the floor says are fine.** Broken out:

```
l3-21  cand-6b313c39   0.958919 / 0.978946  vs floor 0.955360   ABOVE   (4 rejections)
l3-21  cand-89fa74fe   0.977751 / 0.978934  vs floor 0.955360   ABOVE   (4 rejections)
l3-43  cand-90886b3c   0.986913             vs floor 0.976682   ABOVE   (live, this run)
l3-48  cand-61f768c8   0.984610             vs floor 0.977767   ABOVE   (minimal witness)
```

`cand-6b313c39` and `cand-89fa74fe` are the two candidates whose empty families ended
`run-l3-21-20260905-071312` about 10 hours early
(`finding-run-stops-with-budget-unused.md`). Both were above the floor on every rejection. So the
gate's cost on that run was not four wasted repair attempts — it was the run.

At the trial level the same comparison gives **118 above the floor / 123 below** across all L3
runs (and 118 / 36 within this run alone, which is where the above-floor cases concentrate). Trial
counts should not be read as distinct losses — they are dominated by a handful of candidates
re-sampling the same configuration across a 40-trial budget.

## What the numbers say AGAINST changing it, which matters more

18 of 28 rejections are of candidates genuinely below the floor — and **13 of those 18 sit within
0.0021 of it**:

```
l3-48  cand-61f768c8   0.976424  vs floor 0.977767   -0.0013
l3-48  cand-dcf4e7e6   0.976424  vs floor 0.977767   -0.0013
l3-48  cand-8cb745ff   0.976424  vs floor 0.977767   -0.0013   (x3)
l3-48  cand-741c2699   0.976153  vs floor 0.977767   -0.0016   (x3)
l3-48  cand-2136993c   0.976111  vs floor 0.977767   -0.0017   (x2)
l3-48  cand-eed411d8   0.976029  vs floor 0.977767   -0.0017
l3-48  cand-dc4b6fec   0.975956  vs floor 0.977767   -0.0018
l3-21  cand-d31b0474   0.953277  vs floor 0.955360   -0.0021
l3-21  cand-fdb4dac6   0.953277  vs floor 0.955360   -0.0021
l3-21  cand-7dcdbd99   0.953274  vs floor 0.955360   -0.0021
```

Only 2 of 18 are clearly wrong (`cand-cb7be6b4` at −0.0128, `cand-eb910a18` at −0.0581). So a
floor-relative gate has to decide what "at least as consistent as the reference" means to within
~0.002, and thirteen candidates would flip on a tolerance choice of exactly that size. That is the
concrete form of the user's "危险性过大": the change is not "accept the 10 obviously-correct ones",
it is "pick a tolerance that also decides 13 marginal ones", and a wrong choice there admits a
genuinely wrong kernel into a *reported speedup*.

Two further complications the ledger makes visible:

- **`cand-61f768c8` appears on both sides** — 0.984610 above on its minimal witness, 0.976424 below
  on its default. One candidate, two witnesses, opposite verdicts. Any floor-relative rule needs an
  answer, and "accept if either witness clears the floor" is exactly the relaxation that makes the
  guarantee hardest to state.
- **`cand-7dcdbd99` is in the below-floor list at −0.0021** — and it is L3:21's *best kernel ever*
  at 15.5 ms (`opop-v2-l3-21-best-result`). It was rejected once at that margin, repaired, and went
  on to win. So the marginal band is not populated only by junk; it contains the project's best
  L3:21 result, which argues the current gate's strictness is survivable *and* that the band is
  where the interesting candidates live. Both readings are available from the same row, which is
  why this is a decision and not a bug.

## What I am doing

Nothing to the gate. This is the third time evidence has accumulated on it and the user's decision
stands; my job here is to make the cost precise, not to relitigate it.

What is now available that was not before: the **exact ledger**. If the change is ever made, the 10
candidates above and the 18 below are enumerated with their margins, so the decision can be made on
the actual distribution rather than on a claim about it.
`scripts/audit_noise_floor_rejections.py` prints both tables.

One narrower thing that is *not* the gate and might be separable: the repair dispatched at
14:28:39 acts on a candidate whose own failure detail already contains the floor comparison showing
it passes. Suppressing repair when the detail shows the candidate above the floor would cut the
wasted-repair cost without touching the accept/reject decision — the candidate still gets rejected,
it just does not get "fixed". That is a smaller change with a clearer boundary, and I am flagging
it rather than making it, since it still turns on the same 0.002 question.
