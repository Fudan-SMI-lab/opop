# Finding: an optimization can be gated behind a branch the harness never takes

`run-l3-21-20260905-071312`, `cand-c0b3b7cd`. A rewrite advertised a fused
depthwise+BN+ReLU6 Triton kernel, was published, tuned over a full 40-trial budget,
and finished at **25.0 ms — the worst result in the run** (best was 15.50). The fused
kernel **never executed once**. Every trial timed a plain depthwise fallback.

This is worse than a failure, because nothing in the pipeline reports a problem:
correctness passes, timings are produced, the candidate is published, and the event log
shows an ordinary uncompetitive result.

## The mechanism

The candidate branches on the BatchNorm module's mode (`candidate/source.py:124`):

```python
if bn.training:
    _depthwise_kernel[grid](...)          # plain depthwise
    x = self.depthwise_conv[2](bn(depthwise))   # torch BN + ReLU6
else:
    _depthwise_bn_relu6_kernel[grid](...)  # THE ADVERTISED FUSED KERNEL
    x = depthwise
```

The harness **never calls `.eval()` or `.train()`** — verified in both
`KernelBench/src/kernelbench/eval.py` and `gpu/worker_main.py`, neither of which
contains either call. An `nn.Module` defaults to train mode, and
`21_EfficientNetMBConv.py` never switches it. So `bn.training` is **always True**, the
`else` branch is dead code, and the fallback is what gets measured.

Proof from the trial profiles — all 31 completed trials, every one:

```
kernel_names: ["_depthwise_kernel"]
```

Not one launch of `_depthwise_bn_relu6_kernel`.

## It is not the known BN failure — it is the silent cousin

Memory `opop-v2-mbconv-train-mode-bn` records the historical L3:21 failure: using
`running_mean`/`running_var` when the reference needs batch statistics, which produces a
large systematic offset and a **correctness rejection**. That is the loud version.

This candidate does not fail that way, because the `if bn.training:` guard routes the
live path to PyTorch's own `bn()`, which uses batch statistics correctly. The fix for
the loud failure — branch on the mode — *created* the quiet one: the code is correct and
the optimization is unreachable.

## Where it came from: the analyst never saw the run mode

Improvement J already probes this. `SEMANTICS_PROBED` for this run is exactly right:

```json
{"training": true,
 "norm_layers": [{"type":"BatchNorm2d","training":true,"has_running_stats":true,...} x3]}
```

But `eval_semantics` was wired to only **two** of the five agents — `generator` and
`repair`. The **analyst, rewriter and novelty** agents never received it.

So the analyst, reasoning without the mode, proposed (`BOTTLENECK_REPORTED`, 08:07):

> **H1**: "Fuse depthwise convolution, batch-normalization affine transform, and ReLU6
> into one Triton kernel, folding **inference** batch-normalization parameters into the
> depthwise weights and bias."

Its own `risk` field names the problem — "requires inference-mode batch-normalization
semantics" — and its H2 states the correct constraint: "Training-mode batch statistics
require the existing reduction to finish before depthwise consumption." The analyst knew
both modes were possible; it just had no way to know which one applies. The rewriter took
H1, discovered the conflict while writing, and resolved it by adding a mode branch — the
locally reasonable move that strands the optimization.

In TRAIN mode the fold H1 asks for is **impossible in principle**: BN's scale and shift
depend on the current batch's mean/var, which are unknown until the batch has been
reduced, so they cannot be folded into preceding weights. The hypothesis was unbuildable
from the start.

## Cost

| | |
|---|---|
| rewrite agent calls wasted | 1 |
| parameterizer + witness evaluations | 1 published space, 3 knobs |
| tuning trials spent measuring the fallback | **31 completed (a full 40-trial budget)** |
| result | 25.0 ms, worst in the run |

A `fam-5dfc36d7` rewrite round produced this instead of a real candidate.

## Two fixes, both applied

Driver-side, so they take effect on the next run (L3:43 and any rerun), not on the
L3:21 run in flight.

**1. Wire `eval_semantics` to the three agents that lacked it** (`agents/modules.py`,
`control/orchestrator.py`). `AnalystInputs`, `RewriterInputs` and `NoveltyInputs` gained
the field; all three now seed `task/eval_semantics.md` and reference it in their prompts.
The analyst is additionally told that every hypothesis must be executable under the
stated mode, that an inference-BN fold must not be proposed in TRAIN mode, and to state
the mode it assumed in the `risk` field. The rewriter and novelty prompts carry the rule
directly: *the optimized path must be the path that actually executes.*

**2. A static detector** (`paramspace/triton_lint.py`, improvement M).
`_mode_gated_kernel_branches` reports any `if <...>.training:` branch that launches a
Triton kernel on either side, naming the kernels on each side so the agent can see which
half is stranded. It is a **warning, never a hard error** — implementing both modes is
legal — and it reaches the agent through the existing `soft_check` advisory path, so it
can self-correct before the GPU is involved.

### Two false-positive traps, both measured

The first version required the launch counts to be *asymmetric*. It **did not fire on
this candidate**: both sides launch exactly one kernel. Asymmetry is the wrong signal —
the branch existing at all is.

The second version counted every `Subscript` call as a launch, which made
`self.depthwise_conv[2](x)` look like one and fired on **10.8%** of candidates. The
final version resolves module-level `@triton.jit` / `@triton.autotune` names first and
matches only those.

Measured over **213 distinct agent-output kernels** across all runs on disk:

| version | fires | rate |
|---|---|---|
| asymmetric-launch | 0 (misses the true positive) | — |
| any Subscript call | 23 | 10.8% |
| **final** | **1 — exactly this candidate** | **0.5%** |

Of the 33 `.training` branches on disk, **16 launch no Triton kernel on either side** —
the benign pattern of choosing between two torch formulations — and stay silent.

## Scope: L3:21 only, and L3:43 is safe

The mode question only bites where a layer behaves differently in train mode.

- **L3:21** — three `nn.BatchNorm2d` layers, no `.eval()`. Affected. Two other
  candidates also finished at ~25.0 ms launching only a depthwise kernel
  (`cand-080f8c60` seed, `cand-11f83cb7` on the 09-04 run), but **neither has a
  `.training` branch** — they are genuinely unfused, not stranded. This failure mode is
  n=1.
- **L3:43** — two `nn.Dropout` layers in the forward path, which *is* mode-sensitive,
  but both rates are **`attn_pdrop = 0.0`, `resid_pdrop = 0.0`**, so dropout is the
  identity in either mode. Safe.
- **L3:48** — no normalization or dropout layers. Not applicable.

## What this does not claim

n=1 for the stranded-optimization failure. The fix is justified by the mechanism rather
than the frequency: the wiring gap is a certain defect (three agents were structurally
blind to a fact the harness already probed), and the cost when it fires is an entire
candidate budget spent measuring nothing while every signal reads normal.
