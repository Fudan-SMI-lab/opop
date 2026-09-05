# Finding: no behavioral test separates a delegating candidate from a legitimate hybrid

I proposed replacing the blanket `torch.compile` ban with a runtime test: time the candidate
with `torch._dynamo.config.disable` on and off, and reject it if its speed depends on
Inductor. The user asked the right question — would that also reject a legitimate kernel
whose speedup *partly* comes from the compiled graph?

Measured, and the answer is worse than "sometimes": **the test does not work at all.**
Withdrawing the proposal.

## The three shapes that have to be told apart

```
Cheat   compiled graph does 100% of the work; a no-op `_copy` kernel exists for the label
Hybrid  compiled graph accelerates the convs; a REAL `_bnact` kernel does the BN+ReLU6
Honest  a REAL kernel, plain eager torch around it, no compiler anywhere
```

`Hybrid` is exactly the case the question is about, and it is legitimate: the candidate
contract permits leaving torch ops to torch, and 3 of the 4 seeds in the current L3:21 run do
precisely that (with eager torch rather than compiled).

## Attempt 1: the dynamo-disable timing test — unreliable

`dynamo.config.disable = True` makes `torch.compile` fall back to eager silently, so the
ratio OFF/ON should expose a candidate that leans on Inductor. On one workload it looked
convincing (1.22×). It does not replicate:

```
workload               OFF/ON ratios (3 repeats)     verdict
96ch x4 deep           1.39x 1.40x 1.41x             DETECTED
576ch x2               1.07x 1.07x 1.06x             DETECTED
96ch x1 (shallow)      1.13x 0.97x 0.98x             INCONSISTENT
320ch x6 deep          1.23x 1.27x 1.27x             DETECTED
```

The shallow workload — where the compiled graph has least to fuse, i.e. where the cheat is
*cheapest to commit* — is where detection fails, and it fails non-deterministically. A guard
whose sensitivity depends on how much Inductor happened to gain is not a guard.

And on the realistic MBConv shape the separation inverts:

```
                dynamo ON   dynamo OFF   OFF/ON
Cheat              2.73ms      2.73ms     1.00x   <- cheat looks CLEAN
Hybrid             2.70ms      2.43ms     0.90x   <- legitimate looks GUILTY
Honest             2.37ms      2.38ms     1.00x
```

`Hybrid` — the legitimate shape — is the one flagged, and `Cheat` passes. Exactly the
inversion the user was asking about, and it is the common case, not a corner.

## Attempt 2: profile who owns the GPU time — reliable, and still cannot decide

`torch.profiler` sees every launch including Inductor's generated Triton, so ownership of
GPU time is measurable and stable:

```
              our kernel's share      Inductor's share
Cheat          7.7%  (_copy)              92.3%
Hybrid         7.9%  (_bnact)             92.1%
```

**7.7% vs 7.9%.** The two shapes are indistinguishable by time share, because the cheat's
no-op copy touches the same number of bytes the real epilogue does. The only difference is
*semantic* — whether the arithmetic inside the kernel means anything — and that is not
decidable from a profile, a timing, or an AST.

## Why the blanket ban is the right call anyway

The question "does the candidate's kernel do the work, or is it decoration" has no
mechanical answer. Given that, the options are to allow the shape (and accept that the
observed hack passes) or to forbid it. Forbidding costs:

- **0 of 158 legitimate candidates on disk** (the 2 violations are both in the abandoned run)
- **0 of 270 KernelBench references** use any torch compiler
- the rejected shape has a **trivial legal rewrite**: drop `torch.compile`, keep the kernel,
  leave the rest to eager torch. That is what `Honest` is, and on this shape it was the
  *fastest* of the three (2.37 ms vs 2.70/2.73).

So the "legitimate hybrid" being excluded loses access to Inductor's fusion, not to
correctness or to the search. And the harness's own baseline is `torch.compile(reference)` —
a candidate calling `torch.compile` on essentially the reference computation is being
compared with itself, which is the integrity problem regardless of whether a real kernel
sits beside it.

One thing worth stating plainly: the observed cheat was **not even fast**. At 18.7 ms
against a 16.4 ms `torch_compile_tf32` baseline its honest verdict is **0.877×** — it loses.
The danger was never a flattering number; it was that it became a run's *best* candidate and
would have been reported as that run's outcome.

## What is left as a real gap

Nothing detects a candidate that reaches the compiler by another route (`torch._dynamo.optimize`,
a future API, `torch.export` + compile). The AST check covers `torch.compile`,
`torch.jit.script`, `torch.jit.trace` by name. Widening it is cheap when a new route is
observed; guessing at routes now would add rules with no evidence behind them.

Reproduce: the two probes in this note (dynamo-disable timing across four workloads; torch
profiler ownership on the MBConv shape) against `Cheat`/`Hybrid`/`Honest` fixtures.
