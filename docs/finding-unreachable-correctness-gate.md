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

`cand-dc4b6fec` has been dropped. `cand-51dd1857` (H2), which keeps the original scalar
recurrence inside each chunk, is still in flight and may not hit this. Any report from
this run should state that one rewrite candidate was rejected at 0.976 against a task
noise floor of 0.978 — i.e. its structural headroom is unknown, not disproven.
