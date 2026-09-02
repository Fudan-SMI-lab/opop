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
  Accumulate dot products / reductions in fp32.

## Backend

- Prefer `triton` (`@triton.jit` kernels). CUDA via
  `torch.utils.cpp_extension.load_inline` is allowed if declared.
- torch operations are allowed around the custom kernel(s) (layout, reshaping),
  but the core computation you claim to optimize must run in your kernel.
