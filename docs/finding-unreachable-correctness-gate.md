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

## The run produced a controlled comparison

The rewriter proposed two candidates from the same bottleneck report, in the same family,
against the same diagnosed wall (the serial SEQ dependency). They differ in how much they
reassociate the arithmetic:

| candidate | hypothesis | approach | outcome |
|---|---|---|---|
| cand-dc4b6fec | H1 | chunked parallel prefix, dense work moved to tensor-core `tl.dot` | frac 0.975956 → **rejected**, then damaged to 0.844503 + non-finite across 4 attempts |
| cand-51dd1857 | H2 | two-level scalar scan, **original scalar recurrence kept inside each chunk** | **published first try**, witnesses 4.55ms / 16.7ms |

H2 restructures the sequence-level parallelism but leaves the per-chunk arithmetic order
alone, so it stays within the witnesses' agreement. H1 reassociates inside the chunk and
falls into the 2.2-point gap.

The gate therefore selected on **degree of reassociation, not on correctness**. This is
the bias described above, observed directly rather than argued: of two rewrites attacking
the same wall, the conservative one was admitted and the structurally bolder one was
rejected and then destroyed. A search that systematically keeps the timid half of its
structural proposals is the early-pruning failure mode the paper's problem statement
argues against, arising from the harness rather than from the LLM.

## Options considered

1. **Floor-relative threshold.** Accept if
   `frac >= min(pass_frac, noise_floor - margin)` AND `cosine >= cosine_min`, where
   `noise_floor` is the reference's own ieee-vs-tf32 `frac`, already computed for the
   failure message. Modelled against every rejection in this run at `margin = 0.005`:

   | rejection | frac | outcome |
   |---|---|---|
   | cand-eb910a18 a=1 | 0.912517 | stays rejected |
   | cand-eb910a18 a=2 | 0.912517 | stays rejected (also non-finite) |
   | cand-eb910a18 a=3 | 0.912518 | stays rejected (also non-finite) |
   | cand-dc4b6fec a=0 | 0.975956 | **accepted** |

   The three genuinely-wrong attempts stay rejected on frac alone, and two are
   independently blocked by non-finite output. Only the borderline candidate flips.

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

`cand-dc4b6fec` (H1, chunked parallel prefix with tensor-core `tl.dot`) was rejected at
frac 0.975956 against a task noise floor of 0.977767, then damaged by a repair that had
no real bug to fix (0.843332, then 0.844503 twice, all non-finite) across all four
attempts, and dropped. `cand-51dd1857` (H2) published on its first attempt, confirming
that the gate admitted the conservative rewrite and rejected the bold one.

Any report or paper text drawing on this run must say that the H1 rewrite's structural
headroom is **unknown, not disproven**: it was never measured for latency, because it was
rejected on a correctness threshold the reference itself does not meet. Reporting it as a
failed structural direction would be the exact overclaim the report's own
family-honesty clause was added to prevent.

## A second, independent concern this exposes

Even with the threshold fixed, the repair agent has no way to answer "this difference is
the task's own noise, not a bug". It receives the noise floor in the failure message (it
quoted the 97.6% correctly) but the message frames the situation as a defect to diagnose,
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
