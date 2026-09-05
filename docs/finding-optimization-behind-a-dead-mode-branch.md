# Finding: an optimization can be gated behind a branch the harness never takes

`run-l3-21-20260905-071312`, `cand-c0b3b7cd`. A rewrite advertised a fused
depthwise+BN+ReLU6 Triton kernel, was published, tuned over **two** 40-trial budgets,
and finished at **25.0 ms — the worst result in the run** (best was 15.50). The fused
kernel **never executed once**. All 80 trials timed a plain depthwise fallback.

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

Proof from the trial profiles — all 80 trials, every one:

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

## Cost — twice what I first recorded

I first counted one 40-trial budget. The run then gave it a **second** one: improvement K
saw `NUM_WARPS` at the boundary, expanded it, and re-tuned. Both budgets measured the
same dead path, and both returned exactly 25.0 ms.

```
TUNING_DONE  sp-2f309139  25.0      <- first budget
SPACE_EXPANDED  [{'name': 'NUM_WARPS', 'direction': 'max'}]
TUNING_DONE  sp-155aa1e4  25.0      <- K re-tune, identical result
```

| | |
|---|---|
| rewrite agent calls wasted | 1 |
| parameterizer calls | 2 (initial publish + K expansion) |
| analyst calls spent analyzing a fallback | 2 |
| tuning trials measuring the fallback | **80 — two full budgets** |
| result | 25.0 ms both times, worst in the run |

80 of the run's 334 trials — **24% of all GPU tuning work** — went to a kernel that
never ran. The flatness across two budgets and a widened domain is exactly what a dead
path looks like: nothing the tuner varied could matter, because none of it was reached.

This also adds a second entry to improvement K's record: the expansion was not merely
flat (as in `measurement-k-expansion-vs-analyst.md`), it was flat on unreachable code.
K reads `at_boundary` from tuning statistics that were themselves measuring the fallback.

A `fam-5dfc36d7` rewrite round produced this instead of a real candidate.

## Three fixes, all applied

Driver-side, so they take effect on the next run (L3:43 and any rerun), not on the
L3:21 run in flight. The first two are in `44256cd`, the third in `0215b70`.

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

**3. The harness detects a kernel that never ran** (`control/orchestrator.py`).

The first two fixes are preventive; this one is a backstop, and it generalizes past mode
branches to *any* unreachable kernel. It was prompted by watching the run repeat the
mistake: at 08:57 the analyst ran again on this candidate and proposed "fuse the
inference depthwise+BN+ReLU6" a second time — after 31 trials had all timed the fallback.

It could not have known. `TuningStats` has no kernel field and `trials.csv` did not carry
one, so the single fact that settles the question was structurally unavailable to it,
even though `TrialRecord.profile.kernel_names` held it all along.

`_unlaunched_kernels` compares the candidate's defined `@triton.jit` names against every
trial's `profile.kernel_names` and appends a `KERNELS_NEVER_LAUNCHED` event. Being
deterministic, it is journalled as fact rather than left to an agent's judgement — and the
analyst now receives the list, seeded as `tuning/never_launched_kernels.md`, with an
instruction to address it before any resource analysis, since an unreached kernel is not
a slow kernel. `trials.csv` gained a `kernels_launched` column.

One guard matters: it returns nothing when **no** trial carries kernel names (a CUDA
backend, or profiling unavailable), so absence of data never reads as absence of
launches. Verified by replay over real trial data — it names
`_depthwise_bn_relu6_kernel` on `cand-c0b3b7cd` and stays silent on `cand-7dcdbd99` and
`cand-fdb4dac6`, whose kernels all ran.

## Scope: L3:21 only, and L3:43 is safe

The mode question only bites where a layer behaves differently in train mode.

- **L3:21** — three `nn.BatchNorm2d` layers, no `.eval()`. Affected, and **n=3, not the
  n=1 I first reported** (see the audit below). Two ~25.0 ms candidates that launch only
  a depthwise kernel (`cand-080f8c60` seed, `cand-11f83cb7` on the 09-04 run) are *not*
  instances: neither has a `.training` branch, so they are genuinely unfused rather than
  stranded.
- **L3:43** — two `nn.Dropout` layers in the forward path, which *is* mode-sensitive,
  but both rates are **`attn_pdrop = 0.0`, `resid_pdrop = 0.0`**, so dropout is the
  identity in either mode. Safe.
- **L3:48** — no normalization or dropout layers. Not applicable.

## The audit: n=3, 156 trials, and the static lint alone would miss two of them

`scripts/audit_dead_kernels.py` runs the same defined-vs-launched comparison over any set
of runs. Across all 19 runs on disk, **89 tuned candidates** carry kernel metadata and
**3** spent a budget with a kernel that never ran:

| run | candidate | never launched | trials | best ms | static lint |
|---|---|---|---|---|---|
| 09-04 L3:21 | cand-80665a49 | `_pointwise_eval_epilogue_kernel` | 38 | 20.7 | **blind** |
| 09-04 L3:21 | cand-faa71ba0 | `_pointwise_eval_epilogue_kernel` | 38 | 21.1 | **blind** |
| 09-05 L3:21 | cand-c0b3b7cd | `_depthwise_bn_relu6_kernel` | 80 | 25.0 | fires |

**156 trials total.** The two 09-04 cases predate this investigation by a day and were
never noticed — exactly as the failure mode predicts, since both produced plausible
mid-field results (20.7 and 21.1 ms against a 20.5 best).

The `static lint blind` column is the important one. Those two hide the launch inside a
host wrapper:

```python
def _launch_pointwise(x, w, batch_norm=None):
    if batch_norm is None:
        _pointwise_kernel[grid](...)
    else:
        _pointwise_eval_epilogue_kernel[grid](...)   # never reached

# and in forward():
if self.training:
    x = _launch_pointwise(x, w)                       # no kernel[grid] in the branch
else:
    x = _launch_pointwise(x, w, batch_norm=..., ...)
```

The `.training` branch contains no `kernel[grid](...)` at all, so the static check
correctly cannot see it. **2 of 3 are invisible to the static lint and only the runtime
check catches them** — the two are complementary, not redundant, and that is now pinned
by a test.

### One false positive found and fixed by this audit

The first version of the runtime check flagged L3:43 `cand-d257924a`'s `_qk_scores`. That
was wrong: it is a `@triton.jit` **device helper**, called by name (`scores =
_qk_scores(...)`, no `[grid]`) from inside `_softmax_stats` and `_attention_from_stats`.
Triton inlines it, so it produces no separate compiled entry and never appears in
`kernel_names` — while running on all 76 trials. `device_helper_names` now excludes any
jit function invoked as a plain call from another jit body (self-recursion excepted).

Verified in both directions: `_qk_scores` drops out, while `_depthwise_bn_relu6_kernel`
and `_pointwise_eval_epilogue_kernel` are still reported.

## What this does not claim

Three instances, all on one task, is not a frequency estimate. The fixes are justified by
the mechanism rather than the count: the wiring gap was a certain defect (three agents
were structurally blind to a fact the harness already probed), and the cost when it fires
is a whole candidate budget spent measuring nothing while every signal reads normal.

The detectors are also not proof of absence: both need `profile.kernel_names`, so a CUDA
candidate or one whose profiling failed is skipped rather than cleared.

One residual gap I flagged turns out to be empty. A kernel launching on *some* trials but
not the tuned best would pass both checks (the union over trials would contain it). Tested
over all 89 candidates with kernel metadata:

- **0** have a best trial that launched fewer kernels than some other trial;
- **0** have a launched kernel set that varies across trials at all.

Every candidate on disk launches an identical kernel set on every trial, so within this
evidence the union-over-trials comparison is equivalent to a per-trial one. That is a
property of the candidates written so far rather than a guarantee — a candidate could
branch on a knob value — but it means the check is not currently weakened by using the
union.
