# Research: how the state of the art sets correctness tolerance under low-precision reference noise

The question: when the reference itself is imprecise (tf32/bf16), should a candidate be held to
a fixed absolute tolerance, or to a tolerance relative to the reference's own error?

Answered from **primary sources verified on this machine**, not from recollection. Three
distinct approaches exist, and our v2 gate implements none of them correctly.

## 1. KernelBench (the benchmark we use) — tolerance keyed to DECLARED PRECISION

`KernelBench/src/kernelbench/eval.py:83`:

```python
PRECISION_TOLERANCES = {
    torch.float32:  1e-4,   # "By default for fp32, 1e-4 is used according to torchbench"
    torch.float16:  1e-2,   # "torchbench states for bf16 and fp16, use 1e-3 as tolerance
    torch.bfloat16: 1e-2,   #  and 1e-2 if it's too strict"
}
```

applied at `eval.py:804` as `torch.allclose(output, output_new, atol=tol, rtol=tol)` — **every**
element must pass, not 99% of them.

So the upstream benchmark's answer is: **a low-precision computation gets a 100× looser
tolerance than fp32.** Not reference-relative, but precision-relative.

**Measured against our rejections — this is the decisive number.** Taking every distinct
(candidate, dtype) pair we rejected and applying KernelBench's own rule:

```
would pass KernelBench's fp32 tolerance (1e-4):   0 of 27
would pass KernelBench's fp16/bf16 tolerance (1e-2): 27 of 27
```

**All 27 of our rejected candidates are correct by the benchmark's own standard**, because
they declare fp16/bf16 and the benchmark grants low-precision kernels a 1e-2 tolerance. Our
gate rejects them because it applies **one** tolerance (1% per-element, 99% of elements)
regardless of the precision the candidate declares.

## 2. torchbench / `torch._dynamo.utils.same` — an fp64 golden reference, RELATIVE gate

This is the approach that directly answers the question, and PyTorch itself ships it.
`torchbenchmark/util/env_check.py:601` computes a **fp64 golden reference** by casting the
whole model to `torch.float64`:

```python
model_fp64, inputs_fp64 = cast_to(torch.float64, deepcopy_model(model), clone_inputs(...))
fp64_outputs = run_n_iterations(model_fp64, inputs_fp64, ...)
```

and if that fails, falls back to cosine similarity:

```python
except Exception:
    log.warning("fp64 golden ref were not generated for %s. Setting accuracy check to cosine")
    tbmodel.dargs.use_cosine_similarity = True
```

Then `torch/_dynamo/utils.py::same()` first tries plain `allclose`, and **on failure** applies
the relative test:

```python
ref_error = rmse(fp64_ref, ref).item()     # the REFERENCE's own error vs fp64
res_error = rmse(fp64_ref, res).item()     # the CANDIDATE's error vs fp64
multiplier = 3.0 if res.dtype in (torch.float16, torch.bfloat16) else 2.0
passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
```

**That is exactly a noise-floor-relative gate**, and the design choices are all documented in
the source:

- the floor is measured against an **fp64 golden reference**, not against the reference at
  another precision;
- the metric is **RMSE**, a single aggregate — not a per-element fraction;
- the candidate is allowed to be **2× (fp32) or 3× (fp16/bf16) worse** than the reference's
  own error, not merely equal to it. The comment explains the 3.0: *"the end-to-end model's
  accuracy when comparing AMP with FP32 is within a difference of less than 0.1%. Thus it's
  possible that the correctness check failures for these models are false alarms. We use
  multiplier of 3 instead of 2 to avoid these false alarms."*
- the multiplier rises to 8–10× for small tensors, with the reason stated: *"In the presence
  of noise, noise might dominate our error metric for smaller tensors."*

**Verified against our own numbers.** Reproducing the criterion on a 512³ GEMM with MBConv-like
values, fp64 golden reference:

```
reference's OWN rmse vs fp64:  torch@tf32 1.155e-03   <- the floor
                               torch@ieee 9.437e-07

candidate              rmse vs fp64   x floor   dynamo verdict   our gate
fp32 ieee                 1.612e-06      0.00x        PASS         PASS
fp16 + fp32 accumulate    1.155e-03      1.00x        PASS         FAIL   (frac=0.9819)
tf32                      3.106e-03      2.69x        FAIL         FAIL   (frac=0.9807)
bf16                      9.466e-03      8.19x        FAIL         FAIL   (frac=0.8518)
```

The fp16 candidate lands **exactly at 1.00× the tf32 reference's own error** — it is as
accurate as the thing it replaces — and dynamo passes it while our gate fails it. That is the
class of candidate we have been rejecting.

Note also this confirms the fp16-beats-tf32 result independently: 1.155e-03 vs 3.106e-03, so
fp16-with-fp32-accumulation is **2.7× more accurate than tf32** on this shape despite the
identical 10-bit mantissa.

## 3. KernelFoundry (Intel ISL) — where our dual-witness came from, applied DIFFERENTLY

`kernelfoundry/tasks/kernelbench/task.py:131` and `:143` — two **separate** pytest tests:

```python
def test_all_close(...):
    assert all_close_with_slack(out_ref, out_kernel) or all_close_with_slack(out_ref_ieee, out_kernel)

def test_cosine_similarity(...):
    assert cosine_similarity(out_ref, out_kernel) or cosine_similarity(out_ref_ieee, out_kernel)
```

with `all_close_with_slack(max_rel_err=0.01, ratio_below_max_err=0.99)` and
`cosine_similarity(min_sim=0.99985)` — the same constants v2 uses.

The structural difference: KernelFoundry evaluates `(frac on either witness) AND (cosine on
either witness)`, whereas v2 evaluates `(frac AND cosine) on either witness`. **On our data
this changes nothing** — 0 of 279 either way — because:

```
frac passed but cosine failed :   0
cosine passed but frac failed : 279
```

`cosine` passes in **279 of 279** rejections. The gate is effectively single-criterion, and
that criterion is `frac_within_tol`. Worth knowing before anyone tunes `cosine_min`.

## What this means for our gate

Our gate is stricter than all three references, on a specific axis: it applies **one**
per-element tolerance to **every** candidate regardless of declared precision, and it requires
99% of elements to pass rather than testing an aggregate error against a floor.

The three available fixes, in order of how well-supported they are:

1. **Per-precision tolerance (KernelBench's own rule).** A candidate declaring fp16/bf16 gets
   `elem_tol = 1e-2` instead of `1e-2`… note our `relaxed_elem_tol` is *already* 0.01, so this
   is not simply a constant change — KernelBench's `allclose(atol=rtol=1e-2)` is a
   *fundamentally looser* test than ours because `atol + rtol*|ref|` admits large absolute
   error on large elements, whereas our pure relative test does not. This is why all 27 pass
   theirs and none pass ours.
2. **fp64 golden reference + RMSE-relative gate (torchbench/dynamo).** The most principled,
   and the only one that actually measures the floor. Cost: one fp64 forward pass of the
   reference per correctness check — expensive but not prohibitive, and it replaces the
   *second tf32 forward pass we already pay for and never use*
   (`finding-tf32-witness-is-never-the-permissive-one.md`). Risk: fp64 may OOM or be
   unsupported for some ops; dynamo's own fallback for that case is cosine similarity.
3. **Reference-at-another-precision as the floor (what our `floor` metric already computes).**
   Cheapest, already computed on every failure, but it is a *worse* floor than fp64: it
   measures the spread between two imprecise results rather than either one's distance from
   truth.

**None implemented.** Recorded so the decision has primary sources behind it rather than my
reasoning. The key facts for deciding: all 27 rejected candidates pass the upstream
benchmark's own criterion; PyTorch ships a noise-floor-relative gate with a 2–3× multiplier
and documents *why* the multiplier is >1; and the fp16 candidates we reject are measurably as
accurate as the tf32 reference they replace.

Sources, all verified by reading the files on this machine:
- `KernelBench/src/kernelbench/eval.py:83-102, 799-806`
- `/tmp/opencode/pytorch-benchmark-20260811/torchbenchmark/util/env_check.py:515-538, 596-635`
- `<venv>/torch/_dynamo/utils.py::same()`, `rmse()` (torch 2.9.1)
- `kernelfoundry/kernelfoundry/testing.py:33-75`, `kernelfoundry/tasks/kernelbench/task.py:123-146`
