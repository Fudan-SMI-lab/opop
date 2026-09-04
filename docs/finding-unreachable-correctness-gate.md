# Finding: the relaxed correctness gate is unreachable on level3/48

**Status: NOT IMPLEMENTED — needs a decision, because it changes acceptance semantics.**

Found 2026-09-05 during the L3:48 rerun, at the first rewrite round.

## The claim

On `level3/48_Mamba2ReturnY`, a candidate that reassociates the scan's arithmetic is
rejected regardless of whether it is correct, because the gate's `pass_frac = 0.99` is
above what the *reference itself* achieves against its own second witness precision.

## Evidence

`scripts/probe_l3_48_numerics.py`, run on GPU against the reference only — no candidate
involved:

| comparison | frac within 1% | verdict at 0.99 |
|---|---|---|
| reference fp32 vs its own fp64 | 0.999988 | PASS |
| reference fp32 vs torch.compile | 0.999989 | PASS |
| reference fp32 vs **the same code in tf32** | **0.977760** | **FAIL** |

The task's outputs reach 1e22 because `A = nn.Parameter(torch.randn(...))` feeds
`exp(cumsum(A))` over `block_len=64`; `cumsum(A)` spans ±33, so `exp()` spans 3e-15 to
5e14. At that dynamic range the two witness precisions disagree on 2.2% of elements.

The dual-witness gate accepts a candidate matching *either* precision, so in principle a
candidate needs to beat 0.99 against just one. But a candidate whose arithmetic order
differs from the reference lands *between* the two witnesses — and the witnesses are 2.2
points apart, more than the 1 point of slack the gate leaves above 0.99.

`cand-dc4b6fec` is exactly this case. Its H1 rewrite converts the serial recurrence into
chunked parallel prefix — textbook reassociation:

```
vs ieee ref: frac_within_tol=0.975956  cosine=0.99999996  median_rel_err=1.432e-03
vs tf32 ref: frac_within_tol=0.965382  cosine=0.99999991  median_rel_err=1.494e-03
noise floor: frac_within_tol=0.977767  cosine=0.99999996  median_rel_err=3.928e-04
```

Its cosine equals the noise floor's cosine to eight decimals. It is 0.18 points of
`frac` below the floor and was rejected for needing 0.99.

## Why this matters beyond one candidate

This is the same class of defect as the cosine overflow fixed in `fa3ff3f`: a correct
kernel discarded by a gate that cannot be satisfied. It is worse in one respect — the
overflow was a bug with a clear right answer, whereas this is a threshold that is
correct on most tasks and wrong on this one.

It also biases the *experiment*, not just one candidate. The rewriter's most structural
hypotheses are the ones that reassociate arithmetic (chunked prefix, tensor-core GEMMs,
split-K). Those are precisely what an unreachable frac gate filters out, so the search
is pushed toward structurally timid rewrites on exactly the tasks where structural
change matters most. For a paper arguing against premature convergence on a local
structure, silently rejecting the boldest rewrites is a problem in the results, not only
in the plumbing.

## It does not just reject — it actively damages candidates

Observed after the first rejection. `cand-dc4b6fec` was sent to the repair agent, which
diagnosed:

> "The default kernel used 32-element chunks even though the reference performs its
> cumsums and state recurrence in 64-element blocks, changing floating-point accumulation
> boundaries. It also applied TF32 rounding through multiple factorized dot products,
> compounding error enough that only 97.6% of elements matched tolerance."

That diagnosis is **correct**. The agent read the numbers accurately and identified a
real source of numerical difference. But the difference was inside the reference's own
ieee-vs-tf32 spread, so there was no defect to remove — and its fix made the kernel
strictly worse:

| attempt | frac vs ieee | note |
|---|---|---|
| a=0 | 0.975956 | 0.0018 below the noise floor |
| a=1 | **0.843332** | **non-finite output** |

So the gate converted a working, structurally-novel kernel into a broken one. The agent
behaved correctly throughout; it was told there was a correctness bug, and it is not
given the means to answer "this difference is the task's own noise".

Cost per affected candidate: up to `repair_attempts` (3) repair + reparameterize cycles
at roughly 5 minutes of agent wall each. In this run repair is already the single largest
agent cost (0.63h of 1.32h total agent time), and part of that is spent on a candidate
that needed no repair.

### Measured cost, once three families had run

Wall time from each H1 candidate's first event to its last, on `run-l3-48-20260905-010737`:

| candidate | wall | repairs | rejections |
|---|---|---|---|
| cand-dc4b6fec | 39.4 min | 3 | 4 |
| cand-eed411d8 | 17.1 min | 3 | 4 |
| cand-741c2699 | 15.4 min | 3 | 4 |
| **total** | **71.9 min = 1.20h** | **9** | **12** |

All three chains are now complete. Against 3.78h elapsed, that is **32% of the run** spent
on three candidates that were within 0.0019 of the reference's own ieee-vs-tf32 spread when
first rejected — 9 repair calls and 12 rejections, none of which had a defect to fix at
a=0.

The waste is not only the wall time. Repair is the module that times out (36.4%
historically, 2 of the 8 calls in this run), so every unnecessary repair chain is also
extra exposure to a 1200s stall.

### The gate is unreachable even by a *correct* repair

`cand-741c2699`'s first repair did not destroy it (its last one did — see the next
section). The diagnosis — "the kernel used a 16-token temporal tile while the reference
performs its SSD diagonal and state reductions in 64-token chunks" — is right, and the fix
matched the reference's chunking exactly, keeping the output finite:

| | frac vs ieee | below floor | below gate |
|---|---|---|---|
| a=0 | 0.975839 | 0.001928 | 0.014161 |
| a=1 (after repair) | **0.976153** | 0.001614 | 0.013847 |
| noise floor | 0.977767 | — | 0.012233 |
| gate requires | 0.990000 | — | — |

The repair moved frac **+3.14e-4 in the right direction** and was rejected anyway. It closed
16% of the distance to the reference's own agreement level and **2% of the distance to the
gate**.

This is the cleanest statement of the problem available: even when the repair agent
correctly identifies a genuine numerical difference and fixes it without breaking anything,
the target is unreachable, because the remaining 0.0122 gap is *the reference disagreeing
with itself*. No sequence of correct repairs can close it. The other two chains showed that
the loop can make a working kernel worse; this one shows that succeeding does not help
either.

### The three chains exhaust the outcome space

| attempt | cand-dc4b6fec | cand-eed411d8 | cand-741c2699 |
|---|---|---|---|
| a=0 | 0.975956 ok | 0.976029 ok | 0.975839 ok |
| a=1 | 0.843332 **broken** | 0.836301 **broken** | 0.976153 ok |
| a=2 | 0.844503 **broken** | 0.836300 **broken** | 0.976152 ok |
| a=3 | 0.844503 **broken** | 0.836300 **broken** | 0.844827 **broken** |

Two chains collapse on their first repair and stall at a broken fixed point. The third is
the one that makes the point hardest to argue with: it stayed **correct for three
consecutive attempts** — and at a=1 it *improved* frac by +3.14e-4, moving toward the floor
— then broke on a=3, the last attempt before the budget ran out.

So its repair loop spent its entire budget on a kernel that was correct the whole time, made
genuine progress on a real numerical difference, could never reach a threshold 0.0122 beyond
what the reference achieves against itself, and its parting act was to destroy the kernel.

Between the three chains that is every outcome available to a repair loop — break it
immediately, or improve it correctly and still fail, or run out of budget and then break it
— and all twelve terminal states are `rejected`. There is no fourth behaviour left to hope
for, and the loop is not misbehaving in any of them.


## The run produced a controlled comparison — three times, then a fourth instance

In each of three families the rewriter proposed two candidates from the same bottleneck
report, against the same diagnosed wall, differing in how much they reassociate the
arithmetic. The first instance:

| candidate | hypothesis | approach | outcome |
|---|---|---|---|
| cand-dc4b6fec | H1 | chunked parallel prefix, dense work moved to tensor-core `tl.dot` | frac 0.975956 → **rejected**, then damaged to 0.844503 + non-finite across 4 attempts |
| cand-51dd1857 | H2 | two-level scalar scan, **original scalar recurrence kept inside each chunk** | **published first try**, witnesses 4.55ms / 16.7ms |

H2 restructures the sequence-level parallelism but leaves the per-chunk arithmetic order
alone, so it stays within the witnesses' agreement. H1 reassociates inside the chunk and
falls into the 2.2-point gap.

Then it happened again, twice, in unrelated families:

| family | parent | H1 (moves work to tensor cores) | H2 (keeps scalar arithmetic) |
|---|---|---|---|
| fam-99aee6de | 2.09ms | **rejected** at frac 0.975956, damaged to 0.844503 | published, 3.34ms |
| fam-74c41d8d | 2.84ms | **rejected** at frac 0.976029, damaged to 0.836300 | published, **2.46ms** |
| fam-b1ee96ac | 3.80ms | **rejected** at frac 0.975839 | pending |

**3 for 3.** And the three rejected fracs span only **1.9e-4**, each 0.0017–0.0019 below
the 0.977767 floor, with every cosine equal to the floor's own (0.99999996) to eight
decimals. Three independent rewriter calls against three different parent kernels do not
land that close by chance: the frac is a property of *the transformation*, not of any one
candidate's correctness.

The second instance was **pre-registered** — the prediction, with explicit falsifiers, was
committed at `e01708a` before its witnesses ran
(`docs/prediction-gate-selects-on-reassociation.md`). Every part held. So this is no longer
an after-the-fact reading of one pair.

The gate therefore selects on **degree of reassociation, not on correctness**. This is the
bias described above, observed directly rather than argued: of two rewrites attacking the
same wall, the conservative one was admitted and the structurally bolder one was rejected
and then destroyed — every time. A search that systematically keeps the timid half of its
structural proposals is the early-pruning failure mode the paper's problem statement argues
against, arising from the harness rather than from the LLM.

What sharpens it: the analyst named the tensor-core path as the **primary remaining
headroom** in all three families, having ruled out resource limits (80/255 registers, 512 B
of 101,376 shared, one warp, no spills, no OOM). So the harness diagnosed the same
optimisation three times, generated a candidate for it three times, and never timed it
once. The single direction its own analysis points at is the direction the gate
structurally cannot evaluate.

### A fourth instance, and it rules out "one flawed implementation"

Round 2 in `fam-99aee6de` produced `cand-61f768c8`, and it matters because of *why* it
exists. Its change summary opens: "Reformulates the recurrence as a full lower-triangular
contraction **rather than the previously failed chunked scan**". The rewriter read round 1's
failure, discarded that approach, and wrote a structurally different algorithm — a
prefix-log kernel plus on-demand `C@B^T` transition tiles contracted with X, no per-timestep
scalar recurrence at all.

| candidate | family | round | frac | cosine | below floor |
|---|---|---|---|---|---|
| cand-dc4b6fec | fam-99aee6de | 1 | 0.975956 | 0.99999996 | 0.001811 |
| cand-eed411d8 | fam-74c41d8d | 1 | 0.976029 | 0.99999996 | 0.001738 |
| cand-741c2699 | fam-b1ee96ac | 1 | 0.975839 | 0.99999996 | 0.001928 |
| **cand-61f768c8** | fam-99aee6de | **2** | **0.976424** | 0.99999997 | 0.001343 |
| *(noise floor)* | reference | — | *0.977767* | *0.99999996* | — |

Four independent rewriter calls, spread **5.85e-4**, mean 0.976062.

This closes the last alternative explanation. Three instances of one *kind* of rewrite
(chunked scan onto tensor cores) could conceivably share an implementation flaw. This fourth
uses a different algorithm, was written specifically to avoid the first one, and lands in
the same band. What the four have in common is not their structure — it is that they perform
this task's arithmetic in a different order from the reference, using tensor cores.

So the gate does not reject a flawed chunked scan. It rejects **any tensor-core formulation
of this operator**, and the rewriter cannot escape by being more creative: the property being
measured is not a defect it can remove.

#### The fourth diagnosis says there is no bug, in as many words

`cand-61f768c8`'s a=0 repair diagnosis, quoted in full:

> "The dense-scan equations, causal direction, parameter layouts, and initial-state
> contribution **match the reference**, but the default TF32 path rounds both chained dot
> products (C @ B.T and transition @ X). With exponentially scaled transitions and outputs
> up to about 1e22, that compounded input truncation leaves only 97.64% of elements within
> tolerance **despite near-perfect cosine similarity**, causing witness_default_failed."

Every clause is correct, and together they are a finding of *no defect*: the algorithm
matches the reference, the only difference is tf32 rounding at 1e22 dynamic range, and the
cosine is near-perfect. The agent diagnosed the situation more accurately than the gate did.

It then had to produce a fix regardless, because that is the only response the harness
accepts — and the fix took the kernel from 0.976424 to **0.805267 with 23,687,808
non-finite values**, the worst collapse of the four chains.

This is the second concern in this document (a repair channel for "within task noise, no
change needed") reduced to a single case: the agent already knows. It wrote it down. There is
nowhere for that answer to go.



### Half the structural search was never measured

Round 1 completed in all three families. The full yield:

| family | parent | H1 (tensor-core) | H2 (conservative) | net |
|---|---|---|---|---|
| fam-99aee6de | 2.09ms | rejected, then broken | 3.34ms | no gain |
| fam-74c41d8d | 2.84ms | rejected, then broken | **2.46ms** | **improved** |
| fam-b1ee96ac | 3.80ms | rejected, then broken | 3.91ms | no gain |

Six rewrite candidates were generated in round 1. **Three were never measured at all**, and
of the three that were, one improved its family. Round 2 has since added a fourth unmeasured
tensor-core candidate (`cand-61f768c8`), so the count of generated-but-never-timed rewrites
is still climbing.

So the entire measurable output of round 1 across the whole run is a single improvement
(2.84 → 2.46ms) — while the three candidates the analyst ranked as each family's primary
headroom were rejected on a threshold the reference does not meet, and two-thirds of the
rewriter's tokens bought nothing.

This is worth stating carefully in the write-up. It is *not* evidence that the conservative
direction is unproductive — one in three improving is a reasonable hit rate for structural
search, and the paper's thesis is precisely that unpromising-looking branches deserve
rounds. It is evidence that the run's measured hit rate is computed over **half the
candidates generated**, with the omitted half selected not at random but by degree of
reassociation. Any claim about how often structural rewriting helps, drawn from these runs,
is a claim about the conservative half only.


## Options considered

1. **Floor-relative threshold.** Accept if
   `frac >= min(pass_frac, noise_floor - margin)` AND `cosine >= cosine_min` AND the output
   is finite, where `noise_floor` is the reference's own ieee-vs-tf32 `frac`, already
   computed for the failure message. At `margin = 0.005` the effective threshold on this
   task is `min(0.99, 0.977767 - 0.005) = 0.972767`. Modelled against **every** rejection
   the run has produced:

   | rejection | frac | cosine | non-finite | outcome |
   |---|---|---|---|---|
   | cand-eb910a18 a=1 | 0.912517 | nan | no | stays rejected |
   | cand-eb910a18 a=2 | 0.912517 | 1.0 | YES | stays rejected |
   | cand-eb910a18 a=3 | 0.912518 | 1.0 | YES | stays rejected |
   | cand-dc4b6fec a=0 | 0.975956 | 0.99999996 | no | **accepted** |
   | cand-dc4b6fec a=1..3 | 0.843332 / 0.844503 / 0.844503 | 0.99999991 | YES | stays rejected |
   | cand-eed411d8 a=0 | 0.976029 | 0.99999996 | no | **accepted** |
   | cand-eed411d8 a=1..3 | 0.836301 / 0.836300 / 0.836300 | 0.99999995 | YES | stays rejected |
   | cand-741c2699 a=0 | 0.975839 | 0.99999996 | no | **accepted** |

   **3 flip, 9 stay rejected.** Every flip is an a=0 candidate with finite output sitting
   within 0.0019 of the floor; every kernel that stays rejected is either genuinely wrong on
   frac alone (`cand-eb910a18` at 0.9125, 6.5 points below the relaxed threshold) or emits
   millions of NaN/Inf. Non-finite output is an independent hard block, so a broken kernel is
   never admitted however good its frac looks.

   The decisive point is what does *not* appear in that table: with the three a=0 candidates
   accepted, **none of the nine damaged attempts would ever have been generated** — each was
   produced by a repair chain that only started because a=0 was rejected. This option does
   not merely re-admit three kernels; it removes the 1.20h (32% of the run) those chains
   consumed and the two 1200s repair timeouts' worth of exposure that came with them.

2. **Per-task `pass_frac` in the config.** Explicit and auditable, but it is a magic
   number per task and invites tuning the gate until candidates pass — the failure mode
   the anti-reward-hacking work exists to prevent.

3. **Add fp64 as a third witness.** The reference's fp32 matches its own fp64 at
   0.999988, so a candidate could be compared against the trustworthy value instead of
   two mutually-inconsistent fp32 orderings. Costs one extra reference evaluation per
   witness check and materially more VRAM at this tensor size.

4. **Exclude level3/48 from the correctness-comparable task set,** as was already done
   for tasks with `randn` inside `forward` (33/35/41).

## Recommendation

Option 1, with the margin in config and the effective threshold recorded in the event so
every acceptance says which threshold it cleared and why. It is derived from measured
task data rather than hand-set, it demonstrably does not loosen the gate for wrong
kernels, and it degenerates to the current behaviour on tasks whose witnesses agree
(most of them: the floor is ~1.0, so `min(0.99, floor - margin)` is 0.99).

But it is a change to what the harness *calls correct*, and two similar changes were
already deferred for that reason (the dtype-ban, and trying declared alternative values
before rejecting a witness). It should not be made mid-experiment or without a decision.

## Scope for the current run

Three H1 rewrites — one per family, each the analyst's tensor-core hypothesis — were
rejected at frac 0.975956 / 0.976029 / 0.975839 against a task noise floor of 0.977767.
Two were then damaged by repairs that had no real bug to fix (`cand-dc4b6fec` to 0.844503,
`cand-eed411d8` to 0.836300, both with millions of NaN/Inf) across all four attempts and
dropped; the third was still in its chain when this was written. Each family's conservative
H2 sibling published, two of them tuning successfully (3.34ms and 2.46ms).

Any report or paper text drawing on this run must say that these rewrites' structural
headroom is **unknown, not disproven**: none was ever measured for latency, because each was
rejected on a correctness threshold the reference itself does not meet. Reporting them as a
failed structural direction would be the exact overclaim the report's own family-honesty
clause was added to prevent.

Two specific numbers to carry carefully:

- The run's best candidate, **2.46ms** (`cand-a04c3f52`, 7.6x torch.compile on tuned_ms), is
  an H2. It is a real result and the anti-early-pruning fix is what gave its family a
  rewrite round at all — that family began as the third-best seed at 3.55ms. But it must not
  be presented as evidence that the conservative structure is *better* than the tensor-core
  one, only that it is the better of the two the gate allowed to be measured.
- Both that figure and every other speedup here are **lower bounds**, for an unrelated
  reason documented in `docs/finding-one-slow-sample-per-measurement.md`: candidate
  measurements carry one anomalously slow sample per 20 that baseline measurements do not.

## A second, independent concern this exposes

Even with the threshold fixed, the repair agent has no way to answer "this difference is
the task's own noise, not a bug". It receives the noise floor in the failure message (it
quoted the 97.6% correctly, and on `cand-61f768c8` it went further and stated outright that
the equations, causal direction, layouts and initial-state contribution all match the
reference and the cosine is near-perfect — a finding of no defect, in as many words) but the
message frames the situation as a defect to diagnose,
and `_repair_guidance("correctness_mismatch")` tells it to find a numerical error. When
the candidate is at the floor, the honest answer is "no change needed" — and there is no
channel for the agent to say so, nor for the harness to accept that answer. Worth
considering alongside option 1: a repair agent that can return "within task noise, no
change" would have preserved this kernel even under the current threshold.

---

# Adjacent observation: transport timeouts concentrate in the repair module

Not part of the gate finding, but measured while investigating it, and relevant because
repair is now the largest agent cost.

Across the three completed/running L3 runs (151 agent calls total):

| module | ReadTimeouts | calls | rate | clean-call p50 |
|---|---|---|---|---|
| repair | 4 | 11 | **36.4%** | 141s |
| rewriter | 1 | 13 | 7.7% | 191s |
| generator | 0 | 3 | 0% | 460s |
| parameterizer | 0 | 76 | 0% | 98s |
| analyst | 0 | 48 | 0% | 73s |

**What the data rules out.** It is not prompt/context size: repair's sandbox is the
largest (35KB mean) but generator's is comparable (33KB) and generator has never timed
out. It is not slowness: repair's clean calls finish in 141s median, *faster* than both
generator (460s) and rewriter (191s). Repair is bimodal — it either completes quickly or
hangs for the full 1200s. It is not local concurrency: both L3:48 timeouts began with no
other agent call in flight. The `opencode serve` log contains no errors, warnings, or
rate-limit messages, so the stall is upstream of the local server.

**What I could not establish.** Whether repeat repairs on the same candidate are
over-represented. My first attempt to measure this attributed calls to candidates by
"nearest following REPAIR_PRODUCED", which left two of four timeouts unattributed and is
too fragile to support a claim. Testing it properly needs `candidate_id` recorded on
AGENT_CALL_STARTED, which is a one-line change but should not be made mid-run.

**Cost.** 4 × 1200s = 1.33h of pure wall time across the three runs, plus whatever the
retry costs. In L3:43 a single repair call consumed 0.99h — 34% of that run's entire agent
time — because the pre-fix retry logic queued the second attempt behind its own aborted
session. That specific waste is fixed (`995bc96`); the underlying stall is not.

**Not actionable from here.** Lowering `request_timeout_s` would kill legitimate generator
calls (max observed clean call: 478s). The fix that is available — one transport retry on
a fresh session — is already in place and demonstrably works: at both 02:25:11 and
03:36:00 the reset fired and the first one recovered in 4.7 minutes. Worth raising with
whoever owns the model endpoint rather than working around further in the harness.
