# External pure-CUDA kernels vs our Triton output — measured comparison

Three externally-supplied CUDA kernels (`external_files/l3_best_candidates_21_43_48_20260908/`,
sha256 verified against the shipped `manifest.json`) measured against our own best candidates for
the same three KernelBench tasks.

**Everything below was measured on box 2 (RTX 4090), through our own job builders, with the same
relaxed+fp64 correctness gate, 5 correctness trials and 100 CUDA-event samples reported as
medians.** That uniformity is the point: our recorded L3:21 (6.92 ms) and L3:48 (1.41 ms) figures
came from other machines, and the L3:21 one had never been independently re-measured, so quoting
them against numbers taken here would have compared two cards. The reference baseline comes out of
the same job as the candidate, so each speedup is a ratio of two numbers measured in one process.

## 1. Headline: task by task, best against best

| task | external CUDA | ours (Triton) | winner | correctness |
|---|---|---|---|---|
| L3:21 MBConv | 4.833 ms | **4.414 ms** | ours, 1.09x | both 5/5, 0 rescues |
| L3:43 CausalAttention | 8.675 ms (strict fp32) | **3.013 ms** (fp16) | ours, 2.88x | both 5/5, 0 rescues |
| L3:48 Mamba2 | 1.477 ms | **1.411 ms** | ours, 1.05x | ours needed **5/5 fp64 rescues**, theirs 0 |

The L3:43 row is our harness-recorded `final_reeval_ms` of 3.0126 ms for θ_best; re-measuring the
same configuration during this comparison gave 3.029 ms (+0.5%), and bf16 ties it at 3.014 ms.

Two caveats that matter more than the ranking:

- **L3:48 is a tie we should not claim.** 1.411 vs 1.477 is 4.5%, and our kernel passed only via
  the fp64 relative gate on all five trials while theirs passed the primary relaxed gate outright.
  At effectively equal speed theirs is the more accurate kernel. Both tripped the 10x plausibility
  flag (14.6x and 13.5x vs eager) and both were accepted on correctness, as designed.
- **The manifest's own speedups do not reproduce.** Claimed vs measured here, against eager on this
  box: L3:21 3.67x → **3.22x**, L3:43 3.18x → 2.47x, L3:48 **99.2x → 13.99x**. The README warns the
  search-time figures are not GPU0 remeasurements, so this is expected rather than a discrepancy to
  chase — but the 99.2x figure is off by 7x and should not be repeated anywhere.

## 2. L3:43 at matched precision — where CUDA genuinely wins

The overall L3:43 win above is not an apples-to-apples kernel comparison: our candidate computes in
fp16, theirs in fp32. Their file deliberately avoids low precision. Matching precision:

| regime | external CUDA | ours, per-precision retuned | verdict |
|---|---|---|---|
| strict IEEE fp32 | **8.893 ms** | 14.565 ms | **CUDA by 1.64x** |
| tf32 | 5.529 ms | 5.555 ms | **parity** (1.005x) |
| fp16 | — (not implemented) | **3.029 ms** | ours |
| bf16 | — (not implemented) | **3.014 ms** | ours |

Their kernel reaches the tf32 row without any tf32 code of its own: it calls `at::linear` for both
projections, and `at::linear` honours the ambient `torch.backends.cuda.matmul.allow_tf32`. With the
flag off it measures 8.893 ms; with it on, **5.529 ms — a 1.61x gain from a switch outside the
file.** That number matters in §4.

## 3. Splitting the gap: which half of the kernel loses?

Both designs are the same two pieces — two projections (309.2 of the task's 360.8 GFLOP, 85.7%)
plus one fused causal-attention kernel (51.5 GFLOP after causal halving). Timing our GEMM and our
attention kernel separately, and recovering theirs by subtracting standalone cuBLAS from their
total:

| piece | regime | external | ours | gap |
|---|---|---|---|---|
| projections | ieee | 7.551 ms (cuBLAS) | 10.063 ms (best Triton tile) | ours 1.33x slower |
| projections | tf32 | 3.830 ms (cuBLAS) | 4.010 ms | ours 1.05x slower |
| attention | ieee | ≈1.3–1.7 ms *(derived)* | **5.225 ms** | **ours 3–4x slower** |
| attention | tf32 | ≈1.3–1.7 ms *(derived)* | 1.628 ms | parity |
| attention | fp16 | — | 0.831 ms | — |

Their attention figure is derived by subtraction and is **not** directly measurable — their fused
kernel cannot be launched alone. The subtraction gives 1.264 ms from the ieee total and 1.705 ms
from the tf32 total; since their attention is fp32 CUDA-core in both cases it should cost the same
in both, so the 35% spread is the error bar on the method, not a real difference. A plausibility
check supports the range: the attention is 51.5 GFLOP after causal halving, so 1.264–1.705 ms is
**40.8–30.2 TFLOP/s, i.e. 74%–55%** of this card's measured 54.95 TFLOP/s fp32 CUDA-core roof —
high but achievable for a register-blocked kernel. Ours at 5.225 ms is 9.9 TFLOP/s, **18%** of that
roof.

So the answer splits cleanly by regime:

- **At strict IEEE fp32, the loss is in the attention kernel, not the GEMM** (3–4x vs 1.26x).
- **At tf32, there is no loss anywhere** — both pieces are at parity and both totals are ~5.5 ms.

## 4. Is it a Triton limitation, or our tuning?

Separable, and mostly the latter.

**A real Triton weakness (one regime only).** At strict IEEE fp32 our attention kernel reaches 18%
of the fp32 CUDA-core roof against their 55–74%. This is not a tile we failed to find: the whole
36-point tile sweep tops out at 5.225 ms. Triton's `tl.dot(..., input_precision="ieee")` has no
fast path on this hardware — there is no fp32 tensor-core instruction, so it must emulate, whereas
hand-written CUDA issues plain FFMA out of registers. Notably this is *specific to the attention
shape*, not generic: our Triton GEMM at ieee still reaches 56% of the roof (30.7 TFLOP/s). **We
cannot close the ieee attention gap by tuning.** It only matters if strict IEEE is required — and
nothing in these tasks requires it.

**Our search space, which we can fix (three findings).**

1. **One tile for every precision.** Our candidate shipped a single `ATTN_BLOCK_*`/`LINEAR_BLOCK_*`
   set chosen at fp16 (2 bytes/element). At 4 bytes/element that tile needs 131072–164352 bytes of
   shared memory against a 101376 limit, so the shipped configuration **does not even run** at tf32
   or ieee. Retuning per precision moved tf32 from 6.129 → **5.555 ms**, which is exactly where the
   CUDA kernel is. The tf32 "CUDA wins by 1.11x" I reported earlier was this artifact; at parity
   tiles it is 1.005x.
2. **Our Triton GEMM matches cuBLAS wherever it matters.** Swept properly and summed over both
   projection shapes: tf32 1.05x, fp16 1.04x, bf16 1.05x — and in absolute terms 87%, 93% and 93%
   of the respective measured roofs, so both implementations are near the hardware limit and the
   remaining ratio is small. Only strict ieee loses badly, at 1.33x. There is no general "Triton
   can't do GEMM" problem here.
3. **The framework never considers calling the vendor library for a sub-op.** Their kernel's
   projections are simply `at::linear`; ours are hand-written Triton. At ieee that choice alone
   costs us ~2.5 ms. Our candidates *are* permitted to call torch ops (L3:21 candidates used eager
   `F.conv2d` fallbacks), but no contract guidance says *"use cuBLAS where it is already optimal and
   write kernels only where it is not."* That is a generalizable prompt/contract gap, and the
   single most valuable item to come out of this comparison.

**A hypothesis I tested and rejected.** Our `_linear_fwd` lacks the textbook grouped-M L2 swizzle,
normally worth 10–25% on Triton GEMMs, and no `GROUP_M` knob existed for the tuner to find. Adding
it measured 0.98–1.01x across all four precisions and both shapes — no effect. With K=768 the
entire B matrix already fits in L2, so there is nothing for the swizzle to recover. Worth recording
so nobody re-proposes it for these shapes.

## 5. Their headline techniques are all reachable from Triton

Inventory of CUDA-specific machinery in `L3_43/best_kernel.cu`: `cp.async` (13 sites),
`__shfl_xor_sync` (2), `cudaStream` (3), `cudaEvent` (11), pinned `cudaHostAlloc` (7),
`__launch_bounds__` (1), `asm volatile` (3), `cudaFuncSetAttribute` (1), `float4` (32).

- **`cp.async` double-buffering is their stated centrepiece — Triton emits it.** Disassembling our
  own kernels, LDGSTS (the SASS instruction `cp.async` lowers to) appears 0/16/24/32 times at
  `num_stages` 1/2/3/4. Same hardware mechanism, reached through a knob our tuner already searches.
- **The stream/event/pinned-memory machinery is correctness scaffolding, not performance**: it runs
  a side-stream check that the mask really is causal and polls it with `cudaEventQuery`. Removing it
  would not slow the kernel down.
- `float4` vectorization and warp-shuffle reductions have direct Triton equivalents.

## 6. The premise error that decides the overall result

The external file justifies its fp32-only design in a "WHY NO TENSOR CORES" comment: on GeForce Ada
the dense TF32 tensor-core rate supposedly *equals* the FP32 CUDA-core rate (both quoted at 82.6
TFLOPS). Our calibration on this box measures **fp32 54.95 vs tf32 89.06 TFLOP/s — 1.62x**, and
fp16 164.4 / bf16 166.7, i.e. 3.0x. Their own kernel then gains **1.61x** the moment `allow_tf32`
is enabled, which is the calibration figure to two digits and directly contradicts the comment.
(Their fp16 figure, 165, matches our 164.4 — only the tf32 premise is wrong.)

So the largest lever on this task was left unused by a false premise, and that — not any coding
technique — is why our candidate wins overall: 85.7% of the FLOPs are projections, which cost 5.63
ms at the fp32 roof, 3.47 ms at tf32, and 1.88 ms at fp16.

## 7. What this changes for the framework

Already addressed by the P1–P4 batch, and this comparison is direct evidence for it:

- **P1 (compile screen + PRUNED instead of FAIL)** targets exactly the failure that hid the fp32
  branch of L3:43's space: shared-memory over-limit configurations were `FAIL`, which Optuna
  *excludes from the TPE model*, so the tuner learned nothing from 180 of 1004 trials and never
  discovered that the fp32 region needs a smaller tile. They are now `PRUNED` and screened before
  launch.
- **P3 (fp16/bf16 ceilings)** stops the mis-scoring that made this candidate look finished:
  re-scoring its 27 verdicts moved θ_best from 142.2% of ceiling (impossible) to 77.0%, and flipped
  18 verdicts from `compute_bound` to `resource_limited` — i.e. from "you are at the roof" to
  "a resource is binding", which is the actionable and correct reading.

Still open, in priority order:

1. **Contract guidance on vendor libraries** — when a sub-op is a plain large GEMM, calling cuBLAS
   is usually optimal and always cheap; kernels should be written for what surrounds it. Generalizes
   across tasks, no hardcoding.
2. **Precision-aware tile domains.** The tuner searches `COMPUTE_DTYPE` and tile size in one space,
   so it *can* learn the interaction, but a tile valid at 2 bytes/element is often infeasible at 4.
   With P1 the infeasible region is now learnable rather than discarded; whether that is sufficient
   should be checked on the L3:43 re-run.
3. **Report per-precision bests, not one θ_best.** Our candidate is unrunnable at two of the four
   precisions its own space declares. A per-precision best would have surfaced that immediately.

## Reproduction

```
# external kernels (wrappers are new; the .cu files are byte-for-byte, sha256-checked in the shim)
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python ext_eval.py'      # L3:43
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python ext_eval2.py'     # L3:21, L3:48
# our candidates, same builder, same box
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python ours_on_box2.py'
# the splits and sweeps
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python split_l3_43.py'   # projections vs attention
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python gemm_gap.py'      # Triton GEMM vs cuBLAS, swizzle
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python attn_and_total.py'
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python best_total.py'    # end-to-end per-precision
```

## A measurement that had to be re-done

`attn_and_total.py` reported the shipped ieee attention tile at 12.339 ms while `split_l3_43.py`
reported 6.737 ms for the same tile — the same kernel, both 100/30-sample medians. Rather than pick
one, I re-measured in a fresh process, three interleaved passes, with the tf32 tile as an in-run
control: **6.713 / 6.723 / 6.724 ms** for the ieee tile and 1.630 ms for the control every time, on
an idle card. The 12.339 was an artifact of that script. The sweep's derived claim that ieee left
"+136% on the table" is therefore void; the real figure is 6.71 → 5.22 ms, +28.6%. The per-precision
retuning conclusion is unaffected — it rests on `best_total.py`, which measures end-to-end.

## Two numbers I had carried wrong, corrected here

Both were caught by re-measuring the derived figures in `verify_claims.py` rather than trusting the
arithmetic I had written down:

- **bf16 is 3.014 ms, not 4.124 ms.** The 4.124 figure came from an earlier session and had never
  been re-measured on this box. bf16 in fact **ties fp16** (3.014 vs 3.029) on this task. That is
  worth knowing on its own: bf16 has fp32's exponent range, so it avoids the overflow class that
  `opop-v2-cosine-overflow-rejects-correct-kernels` records, at no speed cost here.
- **The ieee GEMM gap is 1.33x, not the 1.25–1.29x** I read off the per-shape sweep — the correct
  comparison sums both projection shapes, which weights the large one properly. The ieee projection
  cost we give up to cuBLAS is ~2.5 ms, not ~1.95 ms.
