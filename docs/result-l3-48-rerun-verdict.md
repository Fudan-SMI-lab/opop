# Result: L3:48 rerun — 1.96 ms, 9.49x torch.compile, verified

`run-l3-48-20260905-010737`, finished 2026-09-05 07:12 after **6.08h** of a 12h budget
(the global freeze fired because all four families had exhausted their rewrite rounds, not
on wall clock). 859 events.

## The verdict, on `final_reeval_ms`

| | value |
|---|---|
| best candidate | `cand-c18203b6` (seed, family `fam-99aee6de`) |
| θ_best | `BLOCK_P=64, NUM_WARPS=1, NUM_STAGES=1` |
| `tuned_ms` (20 samples, quick_test) | 2.09 ms |
| **`final_reeval_ms` (100 samples, fresh process)** | **1.96 ms** |
| correctness at re-eval | **5/5 trials on fresh inputs** |

Speedups on the re-eval figure:

| baseline | ms | speedup |
|---|---|---|
| eager | 28.8 | **14.69x** |
| eager_tf32 | 28.3 | 14.44x |
| **torch_compile** | 18.6 | **9.49x** |
| torch_compile_tf32 | 17.9 | 9.13x |

The honest same-precision comparison is **9.49x vs `torch_compile`**: the winning kernel is
pure fp32 (0 `tl.dot`, no low-precision cast), so the ieee baseline is the right denominator
and no tf32 discount applies. `honest_verdict.candidate_precision` reads `"unknown"` because
the detector looks for a precision knob in `PARAMS` and this candidate has none — the value
is a detector gap, not an ambiguity: the source settles it.

## The re-eval was FASTER than the tuned figure, which inverts the usual gap

`opop-v2-reeval-gap-is-the-real-number` records `tuned_ms` running **+1.5% to +6.7%
optimistic** against `final_reeval_ms`. Here it ran **−6.2% pessimistic** (2.09 → 1.96).

The reason is in the sample distribution, straight from the job file:

```
latency_ms = {"mean": 1.96, "std": 0.829, "min": 1.80, "max": 10.20, "n": 100}
```

`max` is **5.7x** `min` and `std` is **42%** of the mean — the one-slow-sample artefact from
`finding-one-slow-sample-per-measurement.md`. If a single sample is ~10.2 ms and the other 99
share the remainder, they average **1.877 ms**. So:

- **1.96 ms is an upper bound on latency**, not a point estimate; steady-state is ~1.88 ms.
- The 100-sample re-eval dilutes one slow sample over 100 draws where the 20-sample
  `tuned_ms` spread it over 20 — which is why the gap inverted on this candidate rather than
  contradicting the earlier finding. Both are the same artefact seen at two sample counts.

Reported figure stays 1.96 / 9.49x: it is what the harness measured, and it errs against the
result rather than for it.

## The 14.4x tripwire fired — and it is legitimate

`excessive_speedup = true`, `excessive_speedup_note`: *"14.4x vs reference (28.204 ms →
1.960 ms) exceeds the 10x plausibility threshold, but all 5/5 correctness trials passed on
fresh inputs; accepted and flagged for review"*. 21 of this candidate's jobs carry the flag.

Reviewed, as the note asks. It is not reward hacking:

**The kernel does the work.** 2805 bytes, one `@triton.jit` kernel, a `tl.range(0, SEQ)` loop
over all 128 timesteps that loads `x`/`b`/`c` every step and stores every output element. No
caching, no `zeros_like`, no `detach`, no first-call guard, no returning an input. Triton
metadata from the re-eval: `_scan_kernel`, **80 registers, 0 spills**, 1 warp — a real
compiled kernel, and 0 spills means the state tile genuinely lives in registers as claimed.

**The speedup has an arithmetic explanation.** At the task's dimensions (batch 2048, seq 128,
heads 8, d_head 64, d_state 16, block_len 64) the reference's `Y_diag` einsum
`bclhn,bcshn,bhcls,bcshp->bclhp` carries **both** `l` and `s` = block_len, so it is quadratic
in block length:

| | work | |
|---|---|---|
| reference `Y_diag` | `b·c·l·s·h·n·p` | 1.374e11 |
| candidate scan | `b·s·h·n·p` | 2.147e9 |
| ratio | | **64x** |

The reference also materializes a 64x64 `segsum` matrix per `(b,h,c)` and `exp()`s it. So the
formulation the candidate replaced is ~64x more arithmetic; the observed 14.4x sits well
inside that, and falls short of 64x because the scan is memory-bound (0 `tl.dot`, so no
tensor-core throughput to trade against). The output element count `b·s·h·d_head` =
**134,217,728** matches the correctness metrics exactly, confirming nothing was elided.

This is the case the speed-guard change was made for
(`opop-v2-speed-guard-must-not-override-correctness`): under the old semantics this kernel —
the run's best result, verified 5/5 — would have been **discarded** as implausible.

## What the run does not show

The 1.96 ms was achieved **without tensor cores**. All nine `tl.dot`-bearing candidates were
rejected by the acceptance path, all seven scalar candidates published
(`result-every-tensor-core-candidate-was-rejected.md`). Whether the tensor-core direction
would have gone faster is **unknown, not disproven** — none was ever timed.

Convergence reporting is also wrong on this run, as it is on every run: all four families
froze `budget_exhausted`, including `fam-99aee6de` at `[2.09, 2.09, 2.09]` — three rounds
without movement, reported as running out of budget
(`finding-converged-stop-kind-is-unreachable.md`). The run is a live instance of that
finding, produced after it was written.

## Fix verification for this run

Everything worker-side that was committed before or during the run reached it and worked:
ext4 venv, B1 prefetch, cosine-overflow fix, transport retry, the speed guard (above),
K expansion, metrics-crash fix, anti-early-pruning. The fp16-witness fallback, the
`change_summary`/`candidate_id` logging, `HYPOTHESES_FAILED`, the device block, and the K
hard-edge filter are all **driver-side and could not apply here** — they take effect from
`run-l3-21-20260905-071312`, which started at 07:13.
