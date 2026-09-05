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

**5. The winning trial's own worker output contains an in-process control.** Reading
`jobs/cand-cb7be6b4-tr-d1701cb3-eval-48dd56ab.out.json` directly:

```
correct                    true          trials_passed  3/3
correctness_mode           "dual_witness_relaxed"
excessive_speedup          false
latency_ms                 {mean 14.2, std 0.41, min 13.0, max 14.5, n 20}
ref_latency_ms             {mean 41.4, std 0.818, min 40.1, max 42.7, n 10, median 41.442}
speedup_vs_ref_in_worker   2.918
```

The worker timed **the reference itself at 41.4 ms in the same process, in the same
trial** — against the independently measured `eager` baseline of **41.6 ms**. Agreement to
0.5% means the timing harness was functioning correctly during exactly this measurement.
A stalled clock, a stale buffer, or a missing synchronize would have distorted the reference
number too, and it did not.

That is the closest thing to a control the trial record offers, and it is why the outlier
reading has weakened considerably: the 2.9x in-worker speedup was measured against a
correctly-timed reference by the same code, in the same process, seconds apart.

Against that: `n_spills: 12` at `n_regs: 255`, where the 30.1 ms config spills nothing.
Spilling usually costs performance, so the fastest config spilling is mildly odd — though
with a 128-row tile the register pressure is expected and the extra tile size can pay for
the spills.

## The analyst reached the same reading independently

Its `BOTTLENECK_REPORTED` (10:11:53) had the same trial data and drew the same conclusion
about the outlier, without being asked about it:

> "The only credible blocked headroom is the flash-attention row tile: the actual best trial
> uses `ATTN_BLOCK_M=128` and `ATTN_NUM_WARPS=8`, reaches 14.2 ms, and simultaneously
> reaches 255 registers/thread with 12 spills. **The next-fastest trial is 29.8 ms, so a
> lower-register configuration is not already as fast.** Shared memory is only 65,536/101,376
> B (64.6%) and the block uses 256/1,024 threads, making register live state, not shared
> memory or threads, the immediate obstacle to scaling the profitable attention tile."

It reported `ATTN_BLOCK_M: increase, blocked_by: registers, predicted_gain 25%` and made H1
"distribute a larger query tile across more consumer warps, targeting `ATTN_BLOCK_M=256`".

That is corroboration of the *mechanism* from a different direction: the 2x gap is explained
by the tile size, and the 12 spills — which I flagged as the odd detail — are the cost of
that tile rather than evidence against it. It also noticed the same thing I did about the
confounds, listing five other knobs whose apparent boundary signals "all lose to the 14.2 ms
configuration at different values, so they are not demonstrated blockers".

Both of its quantitative claims check out against the trial data exactly:

| analyst said | measured |
|---|---|
| "IEEE dot has a 99.4 ms median versus 35.3 ms for fp16" | ieee median **99.4** (n=8), fp16 median **35.3** (n=17) |
| "next-fastest trial is 29.8 ms" | 2nd best = **29.8** (255 regs, 34 spills) |

Note the register story is not monotonic: the 3rd best (30.1 ms) uses 121 registers and
**zero** spills, so spilling is not simply buying speed — the 128-row tile is.

## What would settle it

1. **`final_reeval_ms`** — 100 samples in a fresh process, 5/5 correctness on new inputs.
   This is the number that decides, and on L3:21 the gap ran +1.9% while on L3:48 it ran
   −6.2%, so the direction is not predictable.
2. **A second sample of `ATTN_BLOCK_M=128`.** I first wrote that this would not arrive
   automatically. **Wrong on the first half, right on the second.** A K expansion *did* fire
   at 10:14:23, widening `QKV_BLOCK_K`, `QKV_NUM_WARPS`, `ATTN_BLOCK_N`, `OUT_BLOCK_M` and
   `OUT_NUM_WARPS`, and the re-tune on `sp-db2e9791` did sample `ATTN_BLOCK_M=128` again as
   `tr-ebf1aa16`, reporting 14.2 ms.

   **But that is not an independent measurement.** Its statistics are identical to the first
   trial's to every digit — `14.2, std 0.41, min 13.0, max 14.5, n 20`, same 255 regs, same
   12 spills — and the event carries `reused_measurement: True` with **no job files on
   disk**. It is the measurement cache replaying `tr-d1701cb3`, exactly as designed: the
   expansion left `ATTN_BLOCK_M`'s domain untouched, so the identical parameter vector hits
   the cache instead of the GPU.

   That is correct behaviour and it saves GPU time, but it means **a K re-tune can never
   reproduce a suspicious measurement** — the cache guarantees the same answer. Worth
   remembering when reading any re-tune that "confirms" a prior best.

3. **A genuinely independent 128-tile sample did arrive** — `tr-9c5d8622`, uncached, at
   **19.8 ms** with a *different* surrounding configuration (`ATTN_BLOCK_N=16` vs 64,
   `ATTN_NUM_STAGES=2` vs 1, `QKV_BLOCK_M=64` vs 128, 213 regs, **0 spills**).

   That is the evidence the outlier reading was missing. Two independently measured
   `ATTN_BLOCK_M=128` trials now exist, at 14.2 and 19.8 ms, and **both beat the 29.8 ms
   best of every other tile size**:

   | latency | ATTN_BLOCK_M | measured or cached |
   |---|---|---|
   | **14.2** | **128** | measured |
   | 14.2 | 128 | *cached replay of the above* |
   | **19.8** | **128** | **measured, different config** |
   | 29.8 | 16 | measured |
   | 30.1 … 30.9 | 16 | measured (6 trials) |
   | 31.8 | 32 | measured |

   **Then a third measured 128-tile trial landed at 59.4 ms, and a 64-tile trial at 20.8** —
   so my "it is the tile size" reading was too quick. The full measured picture, uncached
   only:

   | ms | ATTN_BLOCK_M | ATTN_BLOCK_N | WARPS | STAGES | QKV_BLOCK_M | regs | spills |
   |---|---|---|---|---|---|---|---|
   | **14.2** | 128 | 64 | 8 | 1 | 128 | 255 | 12 |
   | 19.8 | 128 | 16 | 8 | 2 | 64 | 213 | 0 |
   | 20.8 | **64** | 64 | 4 | 1 | 64 | 255 | 0 |
   | 29.8 | 16 | 16 | 2 | 1 | 64 | 255 | 34 |
   | 59.4 | **128** | 16 | — | 4 | — | — | — |

   `ATTN_BLOCK_M=128` alone gives 14.2, 19.8 **and** 59.4. `ATTN_BLOCK_M=64` gives 20.8. So
   the tile is necessary-ish but nowhere near sufficient, and `ATTN_NUM_STAGES` does not
   isolate it either — at large tiles, `STAGES=1` spans 14.2 to 99.8.

## Where this actually leaves the number

Across all 31 measured trials on this candidate: best **14.2**, median **50.5**, worst
**455.0**. Exactly three configurations reach sub-21 ms (14.2, 19.8, 20.8). This is a
high-variance, sparsely-rewarding space, not a smooth surface with one lucky draw.

**What is now settled:** the sub-20 ms region is **real and reachable by more than one
configuration**. Three independently measured trials beat the 29.8 ms cluster (14.2, 19.8,
20.8), from three different parameter vectors, with register counts of 255/213/255 and spills
of 12/0/0. A stale-buffer or skipped-work artifact would not reproduce across unrelated knob
settings, so **the "measurement skipped work" hypothesis is out.**

**What is not settled:** whether **14.2 specifically** is right, or whether the honest figure
for this candidate is nearer 19.8–20.8. That single config is still n=1, it is the one with 12
register spills, and no other measured point comes within 5.6 ms of it.

Either way L3:43 would clear its 18.40 ms bar on the 14.2 config and *miss* it at 19.8 —
so this is precisely the case where the distinction matters, and `final_reeval_ms` (100
samples, fresh process, fresh inputs, re-measuring exactly the θ_best config) is the number
to wait for. I will not call this a result before it lands.

## The full tile picture at 1.03h, 72 trials on this candidate

| ATTN_BLOCK_M | trials | fail rate | measured latencies |
|---|---|---|---|
| 16 | 30 | 11/30 = 37% | 29.8 … 394.0 (17 values) |
| 32 | 20 | 12/20 = 60% | 31.8, 38.0, 41.2, 54.5, 58.2, 77.5 |
| 64 | 11 | 7/11 = 64% | **20.8**, 54.4, 64.0, 99.8 |
| 128 | 11 | 7/11 = 64% | **14.2**, **19.8**, 59.4 |

The regularity worth keeping: **larger tiles reach the fast region but fail far more often.**
The three best results in the run all come from tiles ≥64, and so does a ~64% failure rate.
Every fast result is a large tile; not every large tile is fast.

A fourth measured 128-tile sample then landed at **455.0 ms** — so the measured spread at a
*single* tile size is **14.2 to 455.0 ms, a factor of 32**. Whatever the tile contributes, it
explains none of the variance by itself, and I should not have offered it as the explanation
two commits ago on the strength of two samples.

That is a coherent picture of a real optimum in a fragile region, rather than a measurement
artifact — and it is also why the winning config's n=1 is hard to improve on within this
budget: TPE has to land a large tile *and* the right surrounding values, and most attempts at
the former fail outright.

Until then the honest statement is: *a single trial measured 14.2 ms with stable per-sample
timing and physically ordinary throughput; if it holds, it is the first L3:43 kernel to beat
`torch.compile`'s tf32 path.* Not "L3:43 beats its baseline".

## The seed cohort, complete — and it argues the 14.2 is a lone outlier

All four seeds are now tuned (1.54h), so the run's structural coverage can be compared
against every previous L3:43 run's seed phase:

| run | best | 2nd | median | worst | spread | run's final (tuned / re-eval) |
|---|---|---|---|---|---|---|
| 0902-140823 | 29.1 | 39.1 | 41.0 | 45.2 | 1.55× | 29.2 / 30.1 |
| 0903-020233 | 29.3 | 31.6 | 32.2 | 55.7 | 1.90× | 29.3 / 31.1 |
| 0903-145357 | 21.0 | 22.9 | 23.7 | 60.3 | 2.87× | 19.4 / 20.6 |
| 0904-093730 | 19.6 | 20.0 | 20.6 | 23.7 | 1.21× | 17.9 / **19.1** |
| **0905-091705** | **14.2** | 22.5 | 23.0 | 28.0 | 1.97× | *in flight* |

Two readings, and the second is the one I had been underweighting:

**For the run:** its best seed is 27.6% faster than any previous run's best seed, and this is
the first L3:43 cohort whose leader is under 19.6.

**Against the 14.2 being a property of this cohort:** the *other three* seeds (22.5, 23.5,
28.0) are **worse than 0904's entire cohort** (19.6, 20.0, 21.2, 23.7). By median this is the
third-best cohort of five, not the best. So the run is not broadly stronger — it has one
candidate far ahead of a middling field, which is the same shape as the intra-candidate
picture (one trial at 14.2, nothing else within 5.6 ms).

That matters for the pending verdict in two ways. It weakens any "this run is simply better"
explanation for the 14.2, leaving the single candidate to carry it alone. And it means the
0904 run, whose cohort was tighter and better, still only reached 19.1 ms re-eval — so if the
14.2 does not survive re-evaluation, this run's fallback is a 22.5 seed that would land
*behind* 0904's published result.

## Note on the run's health

1.54h of 12h, 4 of 4 seeds tuned, one rejection — correctly, below the floor — repaired and
republished. No `KERNELS_NEVER_LAUNCHED` events. Two K expansions (one flat from cache, one a
genuine +2.1%). The repair timeout at 09:58 recovered at 10:00:07 and its fix survived the
parameterizer intact (16 `tl.dot` preserved, only the dtype default changed).
