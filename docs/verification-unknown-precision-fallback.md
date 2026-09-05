# Verification: `precision: "unknown"` is correct in all 38 cases, and the L3:48 headline holds

Chased a suspected reporting defect and it is not one. Recording it because the suspicion was
specific and the check is reusable — and because the L3:48 9.49× headline depends on it.

## The suspicion

`cand-e9b995d0` on the clean L3:21 rerun tuned to 22.1 ms and classifies as
`precision: "unknown"`. `_honest_verdict` then falls back to the **ieee** baseline:

```
compared_against              torch_compile          22.2 ms
same_precision_speedup        1.0045
beats_same_precision_baseline true
```

Against the *tf32* bar (`torch_compile_tf32`, 16.4 ms) the same kernel is **0.74×** — a 35%
loss reported as a win. If `unknown` were a mis-detection of a tensor-core candidate, the fallback
would be systematically flattering, and `memory: opop-v2-root-cause-no-speedup` is exactly about
comparing against the wrong-precision strawman.

## The check, and why the suspicion was wrong

Two things had to be true for the fallback to be correct, and both are:

**1. Candidates are timed under ieee.** `worker_main.py:631` calls
`_set_matmul_precision("ieee")` immediately before the timed region, with the comment "Time under
ieee precision (honest fp32)". So a candidate's *torch* operations — the convolutions this
candidate leaves to torch, which the contract permits — run at full fp32 while being timed. It
genuinely computes in ieee, so the ieee bar is the right one.

**2. `unknown` never hides a tensor-core candidate.** Checking every candidate on disk that
classifies as `unknown`:

```
candidates classified 'unknown':                                    38
of those containing tl.dot, float16, bfloat16 or tf32 anywhere:      0
```

**Zero misclassifications.** All 38 are genuinely dot-free: elementwise, reduction, or
BN/activation-fusion kernels with no matmul. `_detect_candidate_precision` returns `unknown` only
after failing to find a dtype knob, an `input_precision` literal, a low-precision cast, *and*
`tl.dot` — and on this corpus that conjunction is exactly right.

`cand-e9b995d0` is a legitimate candidate, for the record: one `@triton.jit` kernel fusing
BatchNorm + ReLU6 in fp32 (`.to(tl.float32)` on the load), computing batch statistics with
`torch.var_mean` — correct for TRAIN mode, which is the trap in
`memory: opop-v2-mbconv-train-mode-bn` — and leaving the convolutions to torch. 22.1 ms vs the
22.2 ms ieee bar is an honest 1.0045×. Unimpressive, not wrong.

## The one headline that rests on this

`run-l3-48-20260905-010737`'s reported best was produced under `precision: "unknown"`:

```
cand-c18203b6   tuned 2.09   reeval 1.96
verdict: compared_against torch_compile, same_precision_speedup 9.4898, beats true
speedups: eager 14.69 | eager_tf32 14.44 | torch_compile 9.49 | torch_compile_tf32 9.13
```

Verified: that candidate has **no `tl.dot`, no float16, no bfloat16, no tf32** — one Triton
kernel with `BLOCK_P`/`NUM_WARPS`/`NUM_STAGES` knobs and no precision knob. So `unknown` is the
correct classification and the ieee bar is the right comparison.

And the headline is robust to the choice regardless: `torch_compile` 9.49× versus
`torch_compile_tf32` 9.13× is a **3.9% gap**, because L3:48 is not matmul-bound and its two
baselines are nearly identical. Unlike L3:43 — where the tf32 bar (18.40) is **1.92× faster** than
the ieee bar (35.40) and picking the wrong one would have doubled the reported speedup — nothing
material turns on it here.

## Nothing changed

No fix, because there is no defect: 38 of 38 correct, and the one headline that depends on the
fallback is verified and insensitive to it.

Two things I would treat as real findings if they showed up later, and the check above is how to
see them:

- an `unknown` candidate that **does** contain `tl.dot` — it would be timed against the ieee bar
  while running tensor cores, which is the flattering-strawman case;
- an `unknown` candidate whose task has a **wide** ieee-vs-tf32 baseline gap (L3:43's is 1.92×),
  where the fallback's choice would materially move the number.

Reproduce by re-running the classification sweep in this note over `runs/*/candidates/*/source.py`.
