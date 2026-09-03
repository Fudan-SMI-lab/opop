# Candidate kernel contract

Your output kernel file MUST satisfy every rule below. The harness parses and
rewrites the file mechanically; violations are rejected automatically.

## File structure

1. One Python file, KernelBench solution format.
2. Define `class ModelNew(nn.Module)`:
   - `__init__` accepts exactly the same arguments as the reference `Model.__init__`.
   - `forward` accepts exactly the same inputs (same order/shapes) and returns the
     same output as the reference `Model.forward`.
3. Do NOT define `get_inputs` or `get_init_inputs` — the harness always evaluates
   against the reference's own input factories.
4. Do NOT import or call anything network- or filesystem-related at module scope
   beyond standard imports (torch, triton, math, etc.).

## The PARAMS block (mandatory)

Declare exactly ONE module-level dict literal named `PARAMS` holding every
tunable knob:

```python
PARAMS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "NUM_WARPS": 4,
    "NUM_STAGES": 2,
}
```

Rules:
- Keys are string literals; values are `int`/`float`/`str` literals only
  (no bools, no expressions, no variables).
- EVERY tunable value flows through `PARAMS[...]` — never duplicate a knob as a
  separate constant, default argument, or hard-coded literal.
- The harness tunes by rewriting ONLY this dict's literal values; the rest of the
  file must work unchanged for any legal combination.
- Values must reach the kernel (e.g. passed as `tl.constexpr` arguments, or into
  `num_warps=` / `num_stages=` launch kwargs). A PARAMS entry that changes nothing
  is rejected.

## Correctness and honesty

- fp32 unless the reference uses another dtype; do not silently downcast.
- No caching of outputs across calls, no reading the reference implementation's
  result, no CUDA stream tricks, no patching of timing functions. Static checkers
  and runtime diff-tests will catch these; they fail the candidate immediately.
- Correctness is checked by a diff-test against the reference on its own input
  shapes. The tolerance is tight: under the strict fp32 mode the harness uses
  `torch.allclose(atol=1e-4, rtol=1e-4)` over the WHOLE output tensor (any single
  element out of tolerance fails the candidate). Under the dual-precision mode the
  harness compares against the reference computed at BOTH tf32 and ieee fp32
  precision and accepts if your output matches EITHER (relative error < 1% on
  >99% of elements). Either way, do not assume loose slack — a numerically sloppy
  reduction (e.g. tf32 accumulation over a long dimension) will be rejected.

## Precision and the tensor-core path

For matmul- and convolution-bound kernels, the arithmetic precision of the dot
product is often the single largest lever on latency — larger than block sizes,
warps, or stages. On this GPU, `tl.dot(..., input_precision="ieee")` runs on the
scalar FMA path and leaves the tensor cores idle; `input_precision="tf32"` (or
casting inputs to fp16/bf16 with an fp32 accumulator) dispatches to the tensor
cores and can be roughly 2x faster for the same shapes. `torch.compile` gets its
speed from exactly this — it uses the tf32 tensor-core path by default.

Because of that, you should treat dot-product precision as a first-class design
choice, not an afterthought:

- Under the dual-precision correctness mode (the L3 experiments use it) the
  harness accepts a result that matches the reference computed at **tf32**. A
  tf32 tensor-core kernel is therefore a legal, accepted candidate — prefer it
  for matmul/conv-bound work unless a diff-test shows it drifts out of tolerance.
- Make the dot precision a tunable knob, e.g. `PARAMS["DOT_PRECISION"] = "tf32"`
  with the value threaded into `tl.dot(..., input_precision=PARAMS["DOT_PRECISION"])`,
  so the tuner can compare `"tf32"` against `"ieee"` on real measurements. Keep
  an `"ieee"` fallback reachable so a candidate that genuinely needs full fp32
  can still be expressed.
- Precision applies to the *dot/accumulate* step. Even on the tf32 input path,
  keep the **accumulator** in fp32 (`tl.zeros(..., dtype=tl.float32)`); tf32 only
  reduces the mantissa of the multiply inputs, not the accumulation, and an fp32
  accumulator over a long reduction is what keeps you inside tolerance. A sloppy
  low-precision *accumulator* is the thing that fails the diff-test, not a tf32
  *input*.

## Backend

- Prefer `triton` (`@triton.jit` kernels). CUDA via
  `torch.utils.cpp_extension.load_inline` is allowed if declared.
- torch operations are allowed around the custom kernel(s) (layout, reshaping),
  but the core computation you claim to optimize must run in your kernel.
