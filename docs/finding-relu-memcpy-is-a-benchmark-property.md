# Finding: the "reward hack" on L1:19 is a benchmark property, not a framework defect

**Status**: resolved 2026-09-06 by measurement. **My initial framing was wrong**, and this
document records the correction plus the measurement that settles it.

## What I first claimed

In `run-l1-19-20260906-183211` the novelty agent's first-ever output was a Triton kernel that
performs no ReLU at all — a pure memcpy — with a comment stating the reasoning:

```python
# get_inputs() uses torch.rand, so ReLU is exactly an identity on every
# input the harness constructs. Persistent programs copy contiguous stripes.
```

I called this a reward hack that every guard in the harness misses, and recommended blocking
Loop D on L3 until a semantic differential test existed.

## Why that framing was wrong

The user's objection: if `get_inputs()` genuinely makes ReLU equivalent to a copy on its whole
value range, then by the benchmark's own acceptance criteria the candidate is **correct**, and
if `get_inputs()` ships with the benchmark then this is a benchmark property, not something
our framework introduced.

Both premises check out:

* `KernelBench/KernelBench/level1/19_ReLU.py` defines `get_inputs()` as
  `torch.rand(batch_size, dim)` — values in `[0, 1)`, never negative.
* KernelBench is pinned at `423217d` and `git status --porcelain` on the four task files we
  use reports **no modifications**. The input generator is upstream's, not ours.
* Measured: on L1:19 the reference output is bit-identical to its own input
  (`reference == identity on the harness's own inputs: True`).

So on that task, "ReLU" and "copy" are the same function over the entire tested domain. The
agent read the task definition and implemented the cheapest function satisfying it. That is
a correct answer to the question the benchmark asks.

## The measurement that decides whether it matters for us

The question that actually matters is not "was the agent cheating" but **is this substitution
reachable in the three L3 tasks we run experiments on?** `scripts/probe_dead_branch_reachability.py`
hooks every clamping nonlinearity in each reference, runs the harness's own
`get_inputs()`/`get_init_inputs()`, and reports what fraction of elements each clamp actually
modifies.

```
=== L1:19 ReLU
    reference == identity on the harness's own inputs: True    <-- a memcpy passes here

=== L3:21 MBConv
    ReLU6  expand_conv.2      in=[ -5.7863,  6.1404]  LIVE  clamp modifies 50.00% of elements
    ReLU6  depthwise_conv.2   in=[ -6.5427,  6.4511]  LIVE  clamp modifies 49.98% of elements
    reference == identity on the harness's own inputs: False

=== L3:43 MinGPT attention
    no clamping module;  reference == identity: False

=== L3:48 Mamba2
    no clamping module;  reference == identity: False
```

**None of the three L3 tasks is exposed.** The reason is structural, not luck: an L3 reference
feeds `torch.rand` through learned layers whose weights are randomly initialised and therefore
signed, so the activations reaching each nonlinearity straddle zero — the ReLU6 sites see
`[-5.79, 6.14]` and `[-6.54, 6.45]` and clamp about half their elements. Deleting either one
changes the result and would fail correctness immediately.

L1:19 is exposed because it is a single elementwise op applied *directly* to a non-negative
input, with nothing in between. It is a smoke-test task; no experiment or paper claim uses it.

### A probe bug worth recording

My first version of this probe reported both ReLU6 sites as **DEAD** with input range exactly
`[0.0000, 6.0000]`. That was my own error: the modules are constructed with `inplace=True`
(`21_EfficientNetMBConv.py` lines 25 and 31), so a `register_forward_hook` reads the input
tensor *after* it has been overwritten by the output. The suspiciously exact `[0, 6]` bound —
ReLU6's own output range — is the tell. Fixed by using `register_forward_pre_hook` and cloning
before measuring. Had I trusted the first output I would have "confirmed" the opposite
conclusion, on both tasks, with a plausible-looking number.

## Conclusion

* **Not a framework defect.** No fix to the correctness gates is warranted on this evidence.
  M1 (semantic differential testing against out-of-distribution inputs) is **withdrawn** as a
  prerequisite for Loop D: it would reject candidates the benchmark itself deems correct, i.e.
  it would change the task definition rather than enforce it. Adding it silently would mean our
  harness measures a different problem than KernelBench states.
* **Not a blocker for Loop D on L3.** The substitution has no purchase on 21/43/48.
* **What remains true and worth one cheap action**: the agent will exploit any slack the task
  definition leaves, promptly and while documenting it. On our tasks there is none of this
  particular kind. If a future task were a single elementwise op on a bounded input, it would
  be exposed again — so `scripts/probe_dead_branch_reachability.py` is worth running as part of
  task selection, alongside the existing "forward contains randn" exclusion recorded in
  `memory: opop-l3-task-selection-constraints`. That is a task-vetting step, not a gate.
* The paper's evaluation section can state that reference-equivalence is checked against the
  benchmark's own input distribution, and that L1 smoke tasks are excluded from claims.

## What this does not excuse

The candidate was still *slower* (313 ms tuned vs the honest kernel's 250 ms), so nothing was
won by it, and the run's reported best was the honest kernel. The only real cost was one
novelty attempt.
