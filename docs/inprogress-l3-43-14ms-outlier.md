# In progress: L3:43 reached 14.2 ms — provisionally the first to beat this task's baseline

`run-l3-43-20260905-091705`, `cand-cb7be6b4`, at 0.87h of a 12h budget. **Provisional: this
is `tuned_ms` from a single trial, no `final_reeval_ms` yet, so it is not a verdict.**
Recorded now because the number would be a first for this task and because the evidence for
and against it is worth writing down before it moves.

## The claim, and why it matters

| | ms |
|---|---|
| candidate `tuned_ms` (fp16) | **14.2** |
| `torch_compile_tf32` (its same-precision denominator) | 18.40 |
| `torch_compile` | 35.40 |
| `eager_tf32` | 28.80 |
| `eager` | 41.60 |

At face value that is **1.296x** against the honest same-precision baseline. **No L3:43 run
has ever beaten its baseline** — the five previous runs peaked at 19.1 ms `final_reeval`
(0.9476x), and before that 20.6 (0.8738x), 30.1, 31.1.

It is also the candidate that was rejected below the floor, repaired (`COMPUTE_DTYPE`
default `tf32` → `ieee`), and republished — so the repair loop produced the run's best
result, as it did on L3:21.

## Why I am not reporting this as a result

**The trial is an isolated outlier.** All 25 completed means for this candidate:

```
14.2 | 29.8 30.1 30.4 30.7 30.8 30.9 31.8 | 35.3 38.0 41.1 41.2 | 50.5 54.5 58.2 64.0
     | 77.5 79.9 82.4 86.7 99.0 99.8 | 366.0 387.0 394.0
```

A 2.1x gap to second place with **nothing in between**. That shape is exactly what a
measurement which skipped work looks like, and it is the first thing to rule out.

**n=1 on the winning configuration.** `ATTN_BLOCK_M=128` was sampled twice: once complete
(14.2), once `runtime_error`. Every other value has many samples and a tight cluster:

| ATTN_BLOCK_M | complete trials | best ms |
|---|---|---|
| 16 | 15 | 29.8 |
| 32 | 7 | 31.8 |
| 64 | 2 | 64.0 |
| **128** | **1** | **14.2** |

## What survives scrutiny

**1. Correctness ran.** `status: complete` requires passing `quick_test`'s 3 correctness
trials *before* the worker times anything (`_run_trial` calls correctness first; the worker
times only `if correct and num_perf`). So this is not an untested kernel.

**2. The timing is internally consistent.** `std 0.41, min 13.0, max 14.5, n_samples 20`.
A one-off fluke would show a wide spread or a min far below the mean; 20 samples within
1.5 ms of each other is a stable measurement of *something*.

**3. It is physically plausible — this is the strongest point.** L3:43's arithmetic, from
its own shapes (B=128, T=512, C=768, H=8, head_dim=96):

```
QKV projection   231.9 GFLOP
q@k^T             51.5 GFLOP
att@v             51.5 GFLOP
out projection    77.3 GFLOP
TOTAL            412.3 GFLOP
```

| | implied throughput |
|---|---|
| candidate at 14.2 ms | **29.0 TFLOP/s** |
| the 29.8 ms cluster | 13.8 TFLOP/s |
| `torch_compile_tf32` at 18.4 ms | 22.4 TFLOP/s |
| `eager` at 41.6 ms | 9.9 TFLOP/s |

29.0 TFLOP/s is **~12% of this GPU's fp16 tensor-core peak** and only **1.3x** the
tf32-compile baseline. Nothing here is too fast to be real — unlike L3:48's 14.4x flag,
which needed a FLOP-ratio argument to justify. A larger attention tile
(`ATTN_BLOCK_M=128` against the cluster's 16) plausibly earns 2x by cutting launches and
raising arithmetic intensity, which is the whole point of a FlashAttention-style kernel.

**4. The kernels launched are the real ones.** `_flash_head_major_kernel`,
`_linear_kernel`, `_linear_to_head_major_kernel` — the same three as every other trial, so
this is not the dead-branch failure mode.

Against that: `n_spills: 12` at `n_regs: 255`, where the 30.1 ms config spills nothing.
Spilling usually costs performance, so the fastest config spilling is mildly odd — though
with a 128-row tile the register pressure is expected and the extra tile size can pay for
the spills.

## What would settle it

1. **`final_reeval_ms`** — 100 samples in a fresh process, 5/5 correctness on new inputs.
   This is the number that decides, and on L3:21 the gap ran +1.9% while on L3:48 it ran
   −6.2%, so the direction is not predictable.
2. **A second sample of `ATTN_BLOCK_M=128`.** Improvement K may expand and re-tune this
   space (the analyst is running now), which would independently resample the region. If
   14.2 reproduces across several trials, the outlier reading is wrong and this is simply a
   much better configuration that TPE found late.

Until then the honest statement is: *a single trial measured 14.2 ms with stable per-sample
timing and physically ordinary throughput; if it holds, it is the first L3:43 kernel to beat
`torch.compile`'s tf32 path.* Not "L3:43 beats its baseline".

## Note on the run's health

0.87h of 12h, 78 trials. Two candidates tuned (22.5, 14.2), one rejection — correctly, below
the floor — repaired and republished. No `KERNELS_NEVER_LAUNCHED` events. The repair
timeout at 09:58 recovered at 10:00:07 and its fix survived the parameterizer intact (16
`tl.dot` preserved, only the dtype default changed).
