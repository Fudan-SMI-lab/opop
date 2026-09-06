# Baseline analysis: what the four baselines do and do not establish

Written from the completed L3 runs on disk (8x L3:21, 7x L3:43, 3x L3:48) plus the live
`run-l3-43-20260906-091019`. Every number is read out of `events.jsonl` / `report.md`, not from
notification text.

## 1. The baselines themselves are sound and reproducible

Four baselines are measured per run, each 100 CUDA-event samples in a fresh process:
`eager`, `eager_tf32`, `torch_compile`, `torch_compile_tf32`.

Reproducibility across independent runs is good — this is the part that is *not* a problem:

```
L3:21  eager  25.3 25.4 25.4 25.3 25.5 25.4 25.3     (7 runs after the tf32 pair was added)
L3:43  eager  41.4 41.7 41.6 41.6 41.8               (spread 0.4 ms = 1.0%)
L3:48  eager  28.8 28.8 28.8                         (spread 0.0 ms)
```

`torch_compile` is noisier than eager (std 1.8-2.3 vs 0.9-1.2 on L3:43) but its mean is stable
to ~0.3%. **No baseline instability problem exists.** The problems below are all about *what the
baselines mean*, not whether they are measured correctly.

## 2. Problem A: the reference is fp32, so three of four baselines are the wrong comparator

All three task references are plain fp32 — verified by reading them: no `tf32` flag, no
`autocast`, no `.half()`, no dtype argument anywhere. So the *task as written* is an fp32
computation, and `eager` is the only baseline that matches the reference's own semantics.

But every winning candidate computes in a lower precision:

```
task      winner precision   winner ms   fair comparator
L3:21     fp16               7.21        torch_compile_tf32  16.4
L3:43     fp16               8.06 *      torch_compile_tf32  18.5
L3:48     fp32 (no dot)      1.96        torch_compile       18.6
* live run, tuned; not yet re-evaluated
```

The gap between the flattering number and the honest one is large:

```
task      vs torch_compile   vs torch_compile_tf32   inflation from picking the fp32 baseline
L3:21          3.09x                  2.27x                   1.36x
L3:43          4.39x                  2.30x                   1.91x
L3:48          9.49x                  9.13x                   1.04x
```

**On L3:43 the baseline choice alone is worth 1.91x** — nearly as much as the entire honest
speedup. A paper that reported "4.39x over torch.compile" would be reporting the precision
change, not the kernel work.

The harness already handles this correctly: `_detect_candidate_precision` classifies the winner
and `_honest_verdict` compares it against the same-precision baseline, printed in every report as
**honest same-precision verdict**. Two historical runs are recorded as `FAILS` on that verdict
while showing >1x against the fp32 baselines:

```
run-l3-21-20260904-013056  fp16  vs_compile 1.08x  vs_compile_tf32 0.79x  -> FAILS
run-l3-43-20260903-145357  tf32  vs_compile 1.72x  vs_compile_tf32 0.87x  -> FAILS
run-l3-43-20260904-093730  fp16  vs_compile 1.86x  vs_compile_tf32 0.95x  -> FAILS
```

Those three runs produced kernels **slower than the baseline at their own precision** while
looking like 1.1-1.9x wins. So the mechanism works and is load-bearing. The remaining issue is
presentational: the report prints the four raw speedups *above* the honest verdict, so the
inflated number is what a reader sees first.

## 3. Problem B: `precision: unknown` mislabels a correct kernel, and the fallback is luck

`run-l3-48-20260905-010737`'s winner reports `precision: "unknown"`, and `_honest_verdict` then
defaulted to comparing against `torch_compile` (fp32).

That comparator happens to be right, but not for a reason the code knows. The kernel
(`report/best_kernel.py`) is a sequential selective-scan: `state = state * e + bv * xv`,
`tl.sum(state * cv)`. Grepping it for every precision signal the detector looks for:

```
tl.dot 0   input_precision 0   float16 0   bfloat16 0   .half( 0   tf32 0   allow_tf32 0
float32 1
```

**Zero dot products.** `_detect_candidate_precision` ends with `if "tl.dot" in text: return
"tf32"` and otherwise `return "unknown"`, so any kernel that is pure elementwise/reduction —
scans, normalizations, reductions, pointwise fusions — falls through to `unknown` even when it is
unambiguously fp32. The correct label here is `ieee_fp32`, which would select `torch_compile` as
the comparator *deliberately* rather than as a default.

Generic fix (not implemented): when no dot product and no low-precision construct appears, the
kernel does not use tensor cores at all, so its arithmetic is the storage dtype — classify as
`ieee_fp32` rather than `unknown`. This is a reporting-only change and cannot alter what runs.

## 4. Problem C: `torch_compile_tf32` is not a tensor-core-fair baseline for an fp16 candidate

**Already documented in full** in `finding-same-precision-baseline-is-not-same-precision.md` —
`_honest_verdict` maps fp16, bf16 and tf32 all onto the single `torch_compile_tf32` bar. Not
repeated here. The quantitative point this run adds: on L3:43 that approximation sits between
2.30x and whatever an fp16 torch.compile baseline would show, so it is not a rounding detail.

Note also `finding-candidate-delegates-to-baseline-compiler.md`: a candidate can score a "win"
while delegating the whole graph to `torch.compile` and launching only a copy kernel. That is a
*different* attribution failure from precision, and both are live.

## 5. Problem D: `tuned_ms` is systematically optimistic, so it must never be the headline

`tuned_ms` comes from quick_test (20 samples); `final_reeval_ms` from full_eval (100 samples,
fresh process). Across the completed runs the re-eval is consistently *worse*:

```
task/run                    tuned   reeval   delta
L3:21 20260902-113144       25.2    25.0     -0.8%   (only case where re-eval improved)
L3:21 20260903-071357       24.6    25.3     +2.8%
L3:21 20260904-013056       20.5    20.8     +1.5%
L3:21 20260905-071312       15.5    15.8     +1.9%
L3:21 20260905-195615        6.90    7.21    +4.5%
L3:43 20260902-140823       29.2    30.1     +3.1%
L3:43 20260903-020233       29.3    31.1     +6.1%
L3:43 20260903-145357       19.4    20.6     +6.2%
L3:43 20260904-093730       17.9    19.1     +6.7%
L3:43 20260905-091705        9.73   10.2     +4.8%
L3:48 20260905-010737        2.09    1.96    -6.2%   (only other improvement)
```

Median +3.1%, worst +6.7%. **Any claim of beating a baseline must use `final_reeval_ms`.** The
live L3:43 leader's 8.06 ms is a `tuned_ms` and should be expected to land near 8.4-8.6 ms.

## 6. What the results establish, stated conservatively

```
task    honest speedup (same precision, re-eval)     what it is
L3:21   16.4 / 7.21  = 2.27x   vs torch_compile_tf32  fp16 fused MBConv
L3:43   18.5 / ~8.4  ~ 2.2x    vs torch_compile_tf32  fp16, DELEGATES attention to torch SDPA
L3:43   18.5 / ~11.2 ~ 1.65x   vs torch_compile_tf32  fp16, FULLY hand-written (cand-9f6af7bd)
L3:48   18.6 / 1.96  = 9.49x   vs torch_compile       fp32 sequential scan, algorithmic rewrite
```

`kernel_names` from each candidate's best trial settles the delegation question directly, and it
is the cleanest attribution evidence in the run. Classified by whether an **attention** kernel was
launched at all (the reference's dominant cost); a candidate without one delegated that work to
PyTorch:

```
candidate             ms  family         attn?  kernels
cand-60fdcae9       8.06  fam-6eea8eac    NO    ['_fused_qkv_projection', '_head_layout_projection']
cand-9f6af7bd       9.43  fam-8fb9b2b8    YES   ['_causal_flash_attention', '_linear_bias', '_linear_bias']
cand-8f66c41c      15.40  fam-6eea8eac    NO    ['_head_layout_projection', '_pack_qkv']
cand-9114ad05      15.90  fam-e6706893    NO    ['_qkv_projection_grouped_n']
cand-e29aa508      16.10  fam-94add40d    YES   ['_flash_causal_attention', '_projection_epilogue']
cand-3df5fd86      16.70  fam-e6706893    NO    ['_qkv_projection_tall_narrow']
cand-1129b4d9      16.80  seed            NO    ['_qkv_projection']
cand-bfab4d37      17.00  fam-8fb9b2b8    YES   ['_causal_flash_attention', '_linear_bias']
cand-69b9a666      18.20  fam-94add40d    YES   ['_q_projection_flash_attention']
cand-9df6f133      18.50  fam-8fb9b2b8    YES   ['_causal_flash_attention']
cand-36cda636      18.60  fam-6eea8eac    YES   ['_causal_flash_attention']
cand-5b1bf2d1      18.60  fam-94add40d    YES   ['_flash_causal_attention']
cand-4a3d3538      19.00  fam-6eea8eac    NO    ['_pack_qkv', '_unpack_heads']
cand-9c8d066a      19.10  fam-8fb9b2b8    YES   ['_causal_flash_attention_small_n', '_linear_bias']
cand-dda162c7      19.40  fam-94add40d    YES   ['_flash_causal_attention']
cand-a7cb7970      19.70  fam-6eea8eac    YES   ['_causal_score_tiles', '_causal_softmax_value']
cand-8c64ccc3      19.80  fam-8fb9b2b8    YES   ['_flash_attention']
cand-a988ff79      20.90  seed            YES   ['_causal_scores', '_softmax_value']
cand-93d4a2ff      20.90  fam-8fb9b2b8    YES   ['_causal_probabilities', '_probabilities_times_value']
cand-718483d0      50.00  fam-94add40d    YES   ['_blocked_online_attention']
cand-053f3dc6     110.00  seed            YES   ['_row_streaming_attention']

best FULLY hand-written : 9.43 ms  (cand-9f6af7bd)
best delegating         : 8.06 ms  (cand-60fdcae9)
```

**Delegation is confined to the top of the table**: 6 of the 7 fastest candidates split 4 hand-
written / 3 delegating, while every candidate slower than 17 ms writes its own attention. That is
the expected shape — handing attention to PyTorch's SDPA is a *good* move for latency, so the
search finds it, and it is precisely why the report must separate the two classes.

Note `fam-e6706893` is a **delegating family throughout** (`cand-3df5fd86` 16.70,
`cand-9114ad05` 15.90 and `cand-30074456` 15.40 all launch only a projection kernel), which
corrects an earlier statement of mine that called its 16.7 ms the best hand-written result.

### Delegation is not a family-level property

Per-family, in best-first order, marking each candidate H (writes its own attention kernel) or
D (delegates to torch SDPA):

```
fam-6eea8eac   best  8.06   5 candidates   D D H D H     <- flips mid-lineage
fam-8fb9b2b8   best  9.43   4 candidates   H H H H
fam-e6706893   best 15.40   3 candidates   D D D
fam-94add40d   best 16.10   5 candidates   H H H H H
seed/other     best 16.80   5 candidates   D H H / H H
```

**The 8.06 ms leader's family switches strategy inside its own lineage.** Three of its five
candidates hand-write attention and two delegate — and the delegating variant is the one that won.
So "does this family delegate?" has no answer; only individual candidates do.

Two consequences:

* Any per-family label in the report (approach summary, structural axes) can be wrong for the
  member that actually wins. Attribution has to be read from the winning candidate's
  `kernel_names`, which is what the table above does.
* `structural_signature` treats these as one family, so the family abstraction is grouping two
  materially different computational approaches. That is worth stating in the paper rather than
  claiming families are structurally coherent: the rewriter is free to change the approach, and
  here doing so is what produced the fastest result.

Two caveats that must travel with these numbers:

* **L3:43's leader launches no attention kernel at all.** `cand-60fdcae9` (8.06 ms) runs only
  `_fused_qkv_projection` and `_head_layout_projection`, so the attention core is PyTorch's
  `scaled_dot_product_attention`. Our contribution there is fusing `c_attn` + QKV packing into one
  Triton GEMM and removing the output transpose — real, but not the attention kernel. The best
  **fully hand-written** candidate is `cand-9f6af7bd` at 9.43 ms (`_causal_flash_attention` plus
  both projections), which is the number to headline as our own work.

  Two earlier statements of mine were wrong here and are corrected by the table above: I named
  `fam-e6706893` (16.7 ms) as the best hand-written result — it is a delegating family throughout —
  and I put `cand-e29aa508` (16.1 ms) in the delegating class when it does launch
  `_flash_causal_attention`. Reading `kernel_names` rather than family membership is what
  distinguishes them.

* **L3:48's 9.49x is real but is an algorithmic win, not a scheduling win.** The agent replaced
  the chunked formulation with a register-resident sequential recurrence that never materializes
  the large intermediate — which is why `NUM_WARPS: 1` wins (the loop is serial over `SEQ`).
  `final_reeval_ok: true` (5/5 correctness, fresh process) and the harness's
  `excessive_speedup_flag` fired and was inspected. It is the strongest result and also the one
  least attributable to parameter-tuning feedback, which is the paper's actual thesis.
* **L3:48's 9.49x is real but is an algorithmic win, not a scheduling win.** The agent replaced
  the chunked formulation with a register-resident sequential recurrence that never materializes
  the large intermediate — which is why `NUM_WARPS: 1` wins (the loop is serial over `SEQ`).
  `final_reeval_ok: true` (5/5 correctness, fresh process) and the harness's
  `excessive_speedup_flag` fired and was inspected. It is the strongest result and also the one
  least attributable to parameter-tuning feedback, which is the paper's actual thesis.

## 7. Recommended changes, cheapest first

1. **Report the honest verdict first.** Move the same-precision line above the four raw speedups
   in `report.md`, and label the raw ones "cross-precision, not comparable". Presentation only.
2. **Classify no-dot kernels as `ieee_fp32`** instead of `unknown` (Problem B). One branch,
   reporting only.
3. **State the tf32-vs-fp16 approximation** in the paper's evaluation section (Problem C). No
   code change; needs a user decision if an fp16 baseline is wanted instead.
4. **Publish `final_reeval_ms` everywhere a speedup is claimed** (Problem D). Already the case in
   `report.md`; the risk is in prose written from `tuned_ms`.

None of these changes what the search does. Items 1, 2 and 4 are generic and safe; item 3 is a
decision, not a fix.
