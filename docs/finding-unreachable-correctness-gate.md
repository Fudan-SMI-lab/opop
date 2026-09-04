# Finding: the relaxed correctness gate is unreachable on level3/48

**Status: NOT IMPLEMENTED — needs a decision, because it changes acceptance semantics.**

Found 2026-09-05 during the L3:48 rerun, at the first rewrite round.

> **Partially superseded, 2026-09-05 (later the same run).** This document originally read
> all 27 of the run's `SPACE_REJECTED` events as one phenomenon. They are two.
> `validation.py:158` tests the default witness *first* and returns on the first failure, so
> `witness_minimal_failed` — 17 of the 27 — means the default witness **passed**, and those
> failures are the minimal witness's fp16 corner overflowing on a 1e22 output, not the gate.
> See `finding-minimal-witness-forces-fp16.md`.
>
> What survives unchanged: the six `witness_default_failed` tensor-core rejections at
> frac 0.9758–0.9764 with the cross-family fingerprint, the unreachability arithmetic, and
> `cand-61f768c8`'s frac 0.978034 (above the reference's own 0.977767 floor, with a finite
> output) which is the single strongest datapoint here. What changes: the repair-cost
> attribution. Of the 1.79h of repair triggered by rejections, **1.33h followed the fp16
> corner and 0.46h the gate** — I had reported 0.85h as gate-attributable. The two also have
> different remedies: this one needs a decision, that one does not.

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

### Measured cost: five chains, 40% of the run

Wall time from each tensor-core candidate's first event to its last, on
`run-l3-48-20260905-010737`:

| candidate | wall | repairs | rejections |
|---|---|---|---|
| cand-dc4b6fec | 39.4 min | 3 | 4 |
| cand-eed411d8 | 17.1 min | 3 | 4 |
| cand-741c2699 | 15.4 min | 3 | 4 |
| cand-61f768c8 | 18.8 min | 3 | 4 |
| cand-dcf4e7e6 | 26.8 min | 3 | 4 |
| **total** | **117.6 min = 1.96h** | **15** | **20** |

All five chains are complete. Against 4.95h elapsed, that is **40% of the run** spent on five
candidates that were within 0.0019 of the reference's own ieee-vs-tf32 spread when first
rejected — 15 repair calls and 20 rejections, and not one of them had a defect to fix at
a=0. Every chain used its full `repair_attempts` budget and every chain ended with the
candidate dropped.

By agent time specifically, attributing each repair call to the candidate whose rejection
window contains it: **0.85h of 1.37h of repair wall — 62% — went to these candidates**
(`docs/finding-agent-wall-is-now-the-bottleneck.md` has the per-candidate table).

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

### The chains exhaust the outcome space

| attempt | cand-dc4b6fec | cand-eed411d8 | cand-741c2699 | cand-61f768c8 |
|---|---|---|---|---|
| a=0 | 0.975956 ok | 0.976029 ok | 0.975839 ok | 0.976424 ok |
| a=1 | 0.843332 **broken** | 0.836301 **broken** | 0.976153 ok | 0.805267 **broken** |
| a=2 | 0.844503 **broken** | 0.836300 **broken** | 0.976152 ok | **0.978034 ok — above floor** |
| a=3 | 0.844503 **broken** | 0.836300 **broken** | 0.844827 **broken** | 0.896239 **broken** |

(The fifth chain, `cand-dcf4e7e6`, repeats the first pattern: 0.976424 → 0.805268 →
0.830554 → 0.830555, broken from a=1 onward.)

Every distinct behaviour a repair loop can exhibit is present:

- **break it immediately and stall** (dc4b6fec, eed411d8) — later attempts numerically
  identical to the first broken one, each from a different diagnosis;
- **improve it correctly and still fail** (741c2699 a=1, +3.14e-4 toward the floor), then
  break it when the budget runs out;
- **improve it past every reasonable correctness criterion and then break it anyway**
  (61f768c8 a=2 at 0.978034, above the floor on frac, median, p99 and max-abs-diff alike —
  destroyed at a=3).

All sixteen terminal states are `rejected`. Four chains, twelve repair calls, full budget
each, and every candidate dropped. The loop is not misbehaving in any of them: it is being
asked to reach a threshold 0.0122 beyond what the task's own reference achieves against
itself, and the last chain demonstrates that even *arriving* at the reference's own accuracy
is not enough, because 0.978034 < 0.99.


## The run produced a controlled comparison — five times

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

#### Fifth instance: the fingerprint is reproducible to six decimals across families

`cand-dcf4e7e6` (round 2 of `fam-74c41d8d`, parent `cand-a04c3f52`) is a *different* kernel
from `cand-61f768c8` — different source sha, different structural signature, different
family, different parent — written by a different rewriter call. Its a=0 metrics:

| | cand-61f768c8 | cand-dcf4e7e6 |
|---|---|---|
| frac vs ieee | 0.976424 | **0.976424** |
| median_rel_err | 1.431e-03 | **1.431e-03** |
| p99_rel_err | 2.349e-02 | 2.351e-02 |
| frac vs tf32 | 0.962259 | 0.962258 |
| cosine | 0.99999997 | 0.99999997 |

And the five instances cluster by *algorithm class*, not by candidate:

| candidate | frac vs ieee | median_rel_err | frac vs tf32 | class |
|---|---|---|---|---|
| cand-dc4b6fec | 0.975956 | 1.432e-03 | 0.965382 | chunked scan |
| cand-eed411d8 | 0.976029 | 1.441e-03 | 0.962411 | chunked scan |
| cand-741c2699 | 0.975839 | 1.432e-03 | 0.962538 | chunked scan |
| cand-61f768c8 | **0.976424** | **1.431e-03** | 0.962259 | dense contraction |
| cand-dcf4e7e6 | **0.976424** | **1.431e-03** | 0.962258 | dense contraction |

Two independently written dense-contraction kernels land on the same fingerprint to six
decimal places. That is not a shared bug — two different programs do not reproduce a bug to
six digits. It is the deterministic numerical signature of *doing this operator's arithmetic
that way* on this hardware.

Which settles what the gate is actually measuring. `frac_within_tol` here is a property of
the arithmetic formulation, not of the candidate's correctness: pick a formulation and the
number is determined, whoever writes the kernel. A threshold at 0.99 does not test whether a
kernel is right; on this task it tests *which formulation was chosen*.

#### Final tally: 12 measurements, 4 families, spread 6.5e-4

The run continued past the fifth instance and the fingerprint held every time. Every finite
rejection of a `tl.dot`-bearing candidate, collected at the end (excluding `cand-eb910a18`
at 0.9125, which is a genuinely wrong kernel and the one case that does *not* fit):

| candidate | family | a | frac vs ieee | frac vs tf32 | cosine | class |
|---|---|---|---|---|---|---|
| cand-2136993c | fam-74c41d8d | 0 | 0.975775 | 0.963896 | 0.99999996 | chunked scan |
| cand-741c2699 | fam-b1ee96ac | 0 | 0.975839 | 0.962538 | 0.99999996 | chunked scan |
| cand-dc4b6fec | fam-99aee6de | 0 | 0.975956 | 0.965382 | 0.99999996 | chunked scan |
| cand-eed411d8 | fam-74c41d8d | 0 | 0.976029 | 0.962411 | 0.99999996 | chunked scan |
| cand-2136993c | fam-74c41d8d | 1 | 0.976111 | 0.961993 | 0.99999994 | chunked scan |
| cand-741c2699 | fam-b1ee96ac | 2 | 0.976152 | 0.961405 | 0.99999995 | chunked scan |
| cand-741c2699 | fam-b1ee96ac | 1 | 0.976153 | 0.961405 | 0.99999995 | chunked scan |
| cand-8cb745ff | fam-74c41d8d | 1 | 0.976423 | 0.962258 | 0.99999997 | dense contraction |
| cand-8cb745ff | fam-74c41d8d | 3 | 0.976423 | 0.962258 | 0.99999997 | dense contraction |
| cand-61f768c8 | fam-99aee6de | 0 | **0.976424** | 0.962259 | 0.99999997 | dense contraction |
| cand-dcf4e7e6 | fam-74c41d8d | 0 | **0.976424** | 0.962258 | 0.99999997 | dense contraction |
| cand-8cb745ff | fam-74c41d8d | 0 | **0.976424** | 0.962259 | 0.99999997 | dense contraction |

**n=12, min 0.975775, max 0.976424, spread 6.49e-4 — every value within 0.067% of every
other**, across four families, two algorithm classes, and eight distinct candidates written
by separate rewriter and repair calls.

#### The dense-contraction cluster reaches n=4, and they are provably different programs

`cand-ef6f0748` (06:52, `fam-99aee6de`) made it four. This is the strongest form of the
argument, so the identity check is worth showing in full:

| candidate | family | source sha256 | bytes | `tl.dot` | frac vs ieee |
|---|---|---|---|---|---|
| cand-61f768c8 | fam-99aee6de | ef1b325b… | 11082 | 20 | **0.976424** |
| cand-dcf4e7e6 | fam-74c41d8d | 24ebcba5… | 13787 | 16 | **0.976424** |
| cand-8cb745ff | fam-74c41d8d | 0326e703… | 7350 | 12 | **0.976424** |
| cand-ef6f0748 | fam-99aee6de | be249586… | 9451 | 12 | **0.976424** |

Four distinct source SHAs, four distinct structural signatures, two families, three distinct
parents, four separate rewriter calls — and the sources differ in size by a factor of 1.9.
Each cites a *different* hypothesis text ("full lower-triangular contraction", "materialized
causal semiseparable coefficient matrix", "output-parallel closed-form causal contraction",
"exact closed-form causal operator"). They agree on `frac_within_tol` to six decimal places.

Four independent implementation bugs do not agree to six decimals. `frac_within_tol` is
determined by the *arithmetic formulation* — closed-form causal contraction on this operator
at this precision — and is invariant to who writes it or how.

Every one of the twelve sits **below** the reference's own 0.977767 floor and nowhere near
the 0.99 the gate demands. Both facts follow from the same cause: these are reassociations of
the same arithmetic, and reassociating it costs slightly more than the reference's own
ieee-vs-tf32 disagreement. The gate's threshold is 1.3 percentage points above where any of
them can reach.

The one candidate that *did* clear the floor — `cand-61f768c8` at a=2, frac 0.978034 — is in
the table above and was still rejected. That remains the single strongest datapoint here.

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

#### Then its third attempt beat the reference's own witness agreement — and was rejected

`cand-61f768c8` a=2 is the strongest data point the run has produced, better than anything
option 1 models. Every metric, side by side with the reference compared against *itself*:

| metric | candidate a=2 | reference ieee-vs-tf32 |
|---|---|---|
| frac within tol (vs ieee) | **0.978034** | 0.977767 |
| median_rel_err | **3.926e-04** | 3.928e-04 |
| p99_rel_err | **2.210e-02** | 2.229e-02 |
| max_abs_diff | **3.391e+18** | 3.820e+18 |
| cosine | 0.99999998 | 0.99999996 |

It agrees with the ieee reference **more closely than the reference's two witness precisions
agree with each other**, on every metric — frac, median, p99, and max-abs-diff. Against the
tf32 witness it reaches 0.98461. It was rejected for needing 0.99.

Across every finite rejection in the run this is the only candidate to clear the floor:

```
0.978034  cand-61f768c8 a=2   <== above the noise floor
0.976424  cand-61f768c8 a=0
0.976153  cand-741c2699 a=1
0.976152  cand-741c2699 a=2
0.976029  cand-eed411d8 a=0
0.975956  cand-dc4b6fec a=0
0.975839  cand-741c2699 a=0
0.912517  cand-eb910a18 a=1        (genuinely wrong)
---
0.977767  reference vs its own tf32 witness
0.990000  what the gate demands
```

There is no coherent sense in which this kernel is incorrect. "As accurate as the reference's
own precision variation" is the ceiling for any reimplementation that does not reproduce the
reference's exact operation order — and this candidate reached it, from a tensor-core
algorithm, after a repair chain that was chasing a phantom. The gate's threshold sits 0.0122
above that ceiling.

Note what this also means for option 1: at `margin = 0.005` the effective threshold would be
0.972767, so a=2 would have been accepted — but so would a=0, three attempts earlier, saving
the entire chain.

#### And then the next repair destroyed it

The chain ran to its budget:

| attempt | frac vs ieee | |
|---|---|---|
| a=0 | 0.976424 | finite — the diagnosis said the algorithm matches the reference |
| a=1 | 0.805267 | **broken**, 23,687,808 non-finite |
| a=2 | **0.978034** | finite — **above the noise floor on every metric** |
| a=3 | 0.896239 | **broken**, non-finite |

So the loop reached a kernel provably as accurate as the reference's own precision variation,
was told that was still a failure, and its next repair destroyed it. The a=3 diagnosis is
again correct in substance — it identifies that the candidate used a global prefix and
`(C @ B.T) @ X` where the reference resets cumsums per block and accumulates
`state = (B * decay)^T @ X` then `C @ state`, and calls the difference "reassociation" in as
many words — but it was describing the *only remaining difference* between two
numerically-equivalent formulations, on a kernel that had already cleared the floor.

`cand-61f768c8` was dropped. Its family recorded round 3 with `best_ms` unchanged at 2.09.

This is the fourth chain, and the pattern is now complete in both directions: a repair loop
pointed at a phantom can fail to move a correct kernel, break it immediately, or — as here —
*improve it past the point of any reasonable correctness criterion* and then break it anyway,
because the criterion it is being measured against cannot be reached.





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
   task is `min(0.99, 0.977767 - 0.005) = 0.972767`. Modelled against **all 24 rejections**
   the run produced:

   **Accepted (8)** — every one finite, every one within 0.002 of the floor:
   `dc4b6fec a=0` 0.975956 · `eed411d8 a=0` 0.976029 · `741c2699 a=0/1/2` 0.975839–0.976153 ·
   `61f768c8 a=0` 0.976424 · `61f768c8 a=2` **0.978034** (above the floor) ·
   `dcf4e7e6 a=0` 0.976424

   **Stay rejected (16)** — without exception either genuinely wrong on frac or non-finite:
   `eb910a18` a=0 (no metrics), a=1 0.912517, a=2/a=3 0.9125 + NaN — 6.5 points below the
   relaxed threshold; and the twelve post-repair collapses (0.805–0.896), all with millions
   of NaN/Inf. Non-finite output is an independent hard block, so a broken kernel is never
   admitted however good its frac looks.

   Two things this table makes clear. First, the separation is not marginal: accepted fracs
   span 0.9758–0.9780, rejected ones 0.805–0.9125, with nothing in between — the threshold
   is not being tuned to squeeze candidates through. Second, and decisive: with the five a=0
   candidates accepted, **none of the fifteen post-repair attempts would ever have been
   generated**, because each exists only as a consequence of its a=0 rejection. (Three of
   the eight acceptances above — `741c2699` a=1/a=2 and `61f768c8` a=2 — are likewise
   hypothetical for the same reason.) This option does not merely re-admit five kernels; it
   removes the 1.96h (40% of the run) those chains consumed, the 0.85h of repair agent time
   inside it, and the two 1200s repair-timeout exposures that came with them.

2. **Per-task `pass_frac` in the config.** Explicit and auditable, but it is a magic
   number per task and invites tuning the gate until candidates pass — the failure mode
   the anti-reward-hacking work exists to prevent.

3. **Add fp64 as a third witness.** The reference's fp32 matches its own fp64 at
   0.999988, so a candidate could be compared against the trustworthy value instead of
   two mutually-inconsistent fp32 orderings. Costs one extra reference evaluation per
   witness check and materially more VRAM at this tensor size.

4. **Exclude level3/48 from the correctness-comparable task set,** as was already done
   for tasks with `randn` inside `forward` (33/35/41).

## Scope: does this affect level3/21 and level3/43?

> **FALSIFIED 2026-09-05 07:19, by the first rejection of the L3:21 rerun. The section
> below is wrong and is kept because the reasoning error is the useful part.**
>
> I predicted 21/43 would be unaffected because their outputs are bounded, so the two
> witness precisions would agree to far better than 1%, putting the floor near 1.0. The
> first L3:21 rejection reports its measured floor:
>
> ```
> reference's OWN ieee-vs-tf32 spread: {'frac_within_tol': 0.95536, 'cosine': 0.99999975,
>                                       'ref_absmax': '5.749e+00'}
> ```
>
> **L3:21's floor is 0.95536 — 3.5 points BELOW L3:48's 0.977767, and 4.5 points below the
> 0.99 gate.** `ref_absmax` is 5.749, i.e. thoroughly bounded, exactly as I said. The
> bounded-output premise was correct and the conclusion drawn from it was not.
>
> **The reasoning error:** bounded magnitude does not imply witness agreement. What sets the
> ieee-vs-tf32 spread is how much tf32 mantissa truncation perturbs the *result* — a function
> of reduction depth and cancellation — not of the output's range. L3:48's 1e22 range is why
> its *absolute* differences are 1e19; it is not why its witnesses disagree on 2.2% of
> elements. I conflated the two.
>
> So the gate finding is **not** task-specific in the way this section claims. On the very
> first candidate of the L3:21 rerun, frac 0.958919 vs a 0.95536 floor and cosine 0.99999979
> vs 0.99999975 — **above the floor on both** — rejected for needing 0.99.
>
> What this changes for the decision: Option 1 is a broader change than I represented, and
> the reruns **do** inherit the problem. The `min(0.99, floor - margin)` form matters more,
> not less: on a task with a 0.955 floor it is the difference between a reachable gate and an
> unreachable one. The measured cross-run evidence in
> `result-every-tensor-core-candidate-was-rejected.md` (28 tensor-core candidates published
> on 21/43 vs 0 of 9 on 48) still stands as a fact about *those* runs — but it cannot be
> explained by "the gate does not bite on 21/43", and the honest position is that the
> difference is not yet explained.

#### The cleanest instance in the project so far, from L3:21 (07:22)

`cand-6b313c39`, the same candidate, attempt 1, **fp16** minimal witness — and this one has
no confound at all:

| | frac | cosine |
|---|---|---|
| vs ieee ref | **0.960593** | 0.99999981 |
| vs tf32 ref | **0.978946** | 0.99999994 |
| reference's own floor | 0.955360 | 0.99999975 |
| gate requires | 0.990000 | 0.99985 |

**Above the floor on frac against BOTH witnesses, and above the floor's cosine against both.**
Rejected for needing 0.99.

And this is the **fp16** corner with **zero non-finite values** and `ref_absmax` 5.749 on both
witnesses — so it is not the overflow case from `finding-minimal-witness-forces-fp16.md`. On
L3:21 fp16 is numerically healthy: its 0.978946 against the tf32 reference is *better* than
the default tf32 config managed (0.958919). The more accurate configuration is the one being
thrown away, and the gate is the only thing throwing it.

L3:48's strongest datapoint (`cand-61f768c8` a=2 at 0.978034) needed a caveat: it beat the
floor on the ieee witness only. This one beats it on both, with a cleaner error profile than
the config the harness accepted. It is the single best piece of evidence for the finding and
it comes from a task I had predicted was unaffected.

### The original (incorrect) argument, kept for the record

Checked before the reruns, because if they share the problem their results inherit it. They
do not, and the reason is structural rather than lucky.

| task | what bounds the output | dynamic range |
|---|---|---|
| **level3/48** | *nothing* — `A = nn.Parameter(randn(...))` feeds `exp(cumsum(A))`; cumsum spans ±33 | `exp` spans 3e-15 … 5e14, outputs reach **1e22** |
| level3/43 | `F.softmax(att, dim=-1)` — normalized to [0,1] by construction | bounded |
| level3/21 | conv → `BatchNorm2d` → ReLU6/sigmoid | bounded by BN + activation |

The two witness precisions disagree on 2.2% of level3/48's elements *because* of that
range: at 1e22 a tf32 mantissa and an ieee one land in different relative-error buckets.
Where outputs are O(1), tf32 and ieee agree to far better than 1%, the noise floor sits near
1.0, and `min(0.99, floor - margin)` degenerates to 0.99 — the current behaviour, unchanged.

Confirmed against the previous runs' own rejection messages. Their `correctness_mismatch`
details report max-abs-diff of **0.0014 – 0.004** (L3:43 `run-...-145357`, L3:21
`run-...-013056`), i.e. genuinely small absolute errors on normalized values. L3:48's are
**1.8e19**. Those are not the same kind of failure, and the earlier tasks' rejections look
like real defects rather than gate artefacts.

Two caveats on that comparison. The noise-floor and `ref_absmax` lines were only added at
02:07 and 01:04 today, so the older runs cannot be checked directly — the max-abs-diff
figures are the best available proxy. And L3:21's set includes one 5.66 max-abs-diff, which
is a genuinely broken candidate, not a borderline one.

**Implication for the decision.** Option 1 is not a global loosening: on 21 and 43 it changes
nothing at all, because their floors are ~1.0. It only takes effect on tasks whose reference
cannot meet the threshold against itself, which so far means level3/48 alone. That also
means the reruns of 21 and 43 are unaffected either way, so this decision does not block
them.

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
