# Result: L3:21 reaches 11.4 ms, 1.32× the strongest baseline, independently verified

`run-l3-21-20260905-195615`, 1.8h into 12h. The first rewrite round of `fam-a4a8353c` produced
two siblings; the second one, `cand-f66890d0` (hypothesis H2), tuned to **11.1 ms**.

Timed independently at full sample count in a fresh process, GPU idle, run paused in an agent
call (`scripts/verify_candidate_timing.py`, 100 CUDA-event samples, 20 warmup):

```
  candidate            mean 11.40   median 11.40   min 11.14   max 11.74   std 0.12
  eager (ieee)         mean 25.57   median 25.60                           std 0.31
  eager_tf32           mean 21.27   median 21.26                           std 0.17
  torch_compile        mean 22.56   median 22.56                           std 0.32
  torch_compile_tf32   mean 15.08   median 14.76   min 14.39   max 17.23   std 0.80

vs torch_compile_tf32   1.323x      <- strongest baseline
vs torch_compile        1.979x
vs eager_tf32           1.866x
vs eager                2.243x
```

## Quote 1.32×, not 1.48× — and see the correction below

The run recorded `torch_compile_tf32` at 16.4 ms; I measured **15.08**. That is the one baseline
that disagrees:

```
                       run     mine    delta
eager                 25.3    25.57    +1.1%
eager_tf32            20.9    21.27    +1.8%
torch_compile         22.3    22.56    +1.2%
torch_compile_tf32    16.4    15.08    -8.0%   <- the outlier
```

Three baselines agree within 2%; the compiled tf32 one is 8% *faster* in my measurement, which
makes the comparison **harder** for the candidate. It is also the noisiest of the four — std 0.80
against 0.12–0.32, and a max of 17.23 well above its median of 14.76. So the conservative reading
is mine: **1.32×**, not the 1.48× the run's own numbers would support.

**Correction to the explanation, not the number.** I attributed that 8% to the harness measuring
the baseline too slowly. It does not: the compiled tf32 baseline is **bimodal**, settling at either
~14.7 or ~16.7 ms depending on how long it has been running, with a violent transition between
(`scripts/audit_baseline_warmup_sensitivity.py`; more warmup makes it *slower*, not faster). The
harness's 16.4 is the slow regime and my 15.08 is the fast one — both real. Details in
`finding-rescued-bests-are-tf32-and-tuner-walks-off-them.md`.

This does not weaken the result here; it strengthens the reason to quote the smaller ratio. Against
the fast regime measured directly (14.63 ms), `cand-f66890d0` at 11.40 is **1.28×** — still a win
against the strongest thing torch.compile does on this task, which is the honest bar. Candidates
within ~12% of this baseline cannot be judged from one measurement of either side; this one is not
within 12%.

Separately, `tuned_ms` 11.1 against my 11.40 makes the tuned figure **2.6% optimistic** — inside
the 1.5–6.7% band `opop-v2-reeval-gap-is-the-real-number` documents. That band is behaving as
recorded, and at 2.6% the win survives it comfortably.

## Why it is faster than its 14.7 ms sibling

Both siblings came from the same rewrite call on the same parent. The difference is not tuning:

```
cand-80bf3097  4 kernels   F.conv2d x1 (depthwise stays eager)
   _conv1x1_kernel  _bn_partial_kernel  _bn_finish_kernel  _bn_apply_kernel

cand-f66890d0  5 kernels   F.conv2d x0 (nothing eager left)
   _conv1x1_kernel  _depthwise_fused_input_kernel
   _bn_partial_kernel  _bn_finish_kernel  _bn_final_kernel
```

H2 replaced the eager depthwise convolution with a Triton kernel that **applies the preceding
BatchNorm + ReLU6 while loading its input**, and folds the residual add into the final BN apply.
That removes two full-tensor read-write passes over the largest activation in the block. H1 kept
the depthwise eager and only fused the 1×1 convs, so it still pays those passes. The agent's own
summary said exactly this in advance ("removes intermediate BN apply tensors by normalizing and
applying ReLU6 while loading the following depthwise/project convolution").

This is the paper's claim in miniature: two hypotheses from one rewrite call, the tuner ran both
spaces, and the structurally better one won by 24% — a margin no amount of parameter tuning on
H1 would have closed.

## Correctness is clean and needs no relaxation

5/5 on the **absolute** gate, so no fp64 rescue involved:

```
absolute gate passed 5/5 trials
   vs tf32 ref : frac=0.999691   cosine=1.000000000
   vs ieee ref : frac=0.955402   cosine=0.999999753
   RMSE vs fp64: tf32-ref 7.0296e-04   cand 7.0296e-04   ratio 1.0000
   more accurate than the reference: 4/5 trials
```

`COMPUTE_DTYPE=fp16` with fp32 accumulation, and ratio 1.0000 against the tf32 reference —
matching the `cand-80bf3097` measurement, as expected for two kernels sharing a 10-bit mantissa
and an fp32 accumulator.

No `excessive_speedup` flag fired, correctly: 1.32× is nowhere near the 2× suspicion threshold,
and the mechanism above accounts for the gain without appealing to anything skipped.

## Still open

- **The run's own `final_reeval`**, which is the number that goes in the report. My timing
  predicts it lands near 11.4.
- **`converged`**, still not observed. `fam-a4a8353c` is one rewrite round in and improving
  sharply (19.4 → 11.1), so it should not converge — the signal to watch is a *stalled* family.
- **The other three families**, none of which have had a rewrite round yet.
