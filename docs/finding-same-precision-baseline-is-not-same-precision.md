# Finding: `same_precision_speedup` compares fp16 candidates against a tf32 baseline

`_honest_verdict` exists to stop a tensor-core candidate being reported against the slower
ieee `torch_compile` strawman (`memory: opop-v2-root-cause-no-speedup`). It picks the bar by
a two-way split:

```python
is_tensor_core = precision in ("tf32", "fp16", "bf16")
compile_key = "torch_compile_tf32" if is_tensor_core else "torch_compile"
```

So fp16, bf16 and tf32 all resolve to the **tf32** baseline. But those are three different
precisions: tf32 keeps fp32's 8-bit exponent with a 10-bit significand, fp16 has a 5-bit
exponent, bf16 has 8 exponent bits and only 7 significand bits. A `torch_compile_tf32`
baseline is therefore **not** the same precision as an fp16 candidate.

## Which reported numbers this affects

Every finished L3 run's headline, by the verdict's own fields:

| run | candidate precision | compared against | reported speedup | |
|---|---|---|---|---|
| l3-21 09-04 | fp16 | torch_compile_tf32 | 0.7885 | mismatched |
| l3-21 09-05 | fp16 | torch_compile_tf32 | **1.0316** | mismatched |
| l3-43 09-03 | tf32 | torch_compile_tf32 | 0.8738 | correct |
| l3-43 09-04 | fp16 | torch_compile_tf32 | 0.9476 | mismatched |
| l3-43 09-05 | fp16 | torch_compile_tf32 | **1.8039** | mismatched |
| l3-48 09-05 | unknown | torch_compile | 9.4898 | correct (verified separately) |

**4 of 6, including both of the project's two wins.**

## The direction, which is the saving grace

The mismatch is **conservative**: the candidate computes at a *lower* precision than the
baseline it is credited against. A fair fp16-vs-fp16 comparison would need an fp16 baseline,
which would be **faster** than the tf32 one, so the true same-precision speedup is **lower**
than the number reported — 1.8039 and 1.0316 are upper bounds, not underestimates.

That is the right direction for a gate ("did it beat the bar") but the wrong direction for a
headline: L3:21's **1.0316** is a 3.2% margin over a bar the candidate is not entitled to be
compared against, and it is the run the project cites as its first same-precision win. The
claim may not survive an fp16 baseline.

L3:43's 1.8039 has more room, but the size of the correction is unknown because no fp16
baseline was ever measured.

## Why this is not simply a bug to patch

The verdict cannot report an fp16-vs-fp16 speedup that does not exist: `measure_baseline`
records `eager`, `eager_tf32`, `torch_compile`, `torch_compile_tf32` and nothing else. The
options are all substantive:

1. **Measure an fp16 (and bf16) baseline** — `torch.compile` on the reference cast to fp16.
   Correct, and costs one more 100-sample timing per task per run (~seconds). The catch is
   that the reference at fp16 may not be *numerically valid* for every task — on L3:48 the
   outputs reach 1e22 against fp16's 65504 ceiling, so an fp16 baseline there is meaningless,
   and the harness would have to say so rather than report a number.
2. **Rename the field** to `nearest_available_baseline_speedup` and add
   `baseline_precision`, so the reader sees `fp16 vs tf32` and can judge. Honest, cheap, and
   does not invent a measurement. It weakens the paper's phrasing, which is the point.
3. **Leave it and document** — the mismatch is conservative, so no reported win becomes a
   loss. But "same_precision" in a field name that is not the same precision is exactly the
   kind of label this project has been correcting elsewhere.

**Not implemented.** Option 1+2 together is the technically right answer and it changes the
number every result in the paper quotes, so it is the user's call. Recorded now because the
headline claims rest on it, and because the fix is cheap enough that the only real cost is
re-running the baselines.

## Not to be confused with

- `verification-unknown-precision-fallback.md` — that verifies the *other* branch
  (`unknown` -> ieee bar) is correct, 38 of 38, and L3:48's 9.49× is unaffected because its
  two baselines differ by only 3.9%.
- The tf32-witness finding (`finding-tf32-witness-is-never-the-permissive-one.md`) is about
  the correctness gate, not the timing comparison.

Reproduce: read `honest_verdict.candidate_precision` and `compared_against` from every
`RUN_FINISHED` payload and flag rows where an fp16/bf16 candidate maps to a `_tf32` bar.
