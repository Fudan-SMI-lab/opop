# Triton correctness pitfalls (read before writing any `@triton.jit` kernel)

These are Triton **language / compiler hard constraints** and numerical rules.
Violating them makes the kernel fail to compile or produce wrong results — they
are not style preferences and not performance strategy. Follow every one that
applies. Each shows a BAD form that fails and the GOOD form that works.

## 1. Always mask out-of-bounds loads/stores

```python
# BAD: reads past the end when N is not a multiple of BLOCK_SIZE
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs)                       # out-of-bounds -> garbage / crash

# GOOD: guard every load/store with a mask
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < N
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```

## 2. Block sizes must be compile-time powers of two

```python
# BAD: non-power-of-2 tile — Triton rejects it
offs = tl.arange(0, 100)                         # ERROR

# GOOD: pad to a power of two and mask the tail
BLOCK: tl.constexpr = 128                         # 128/256/512/1024...
offs = tl.arange(0, BLOCK)
mask = offs < 100
```

## 3. Compile-time constants must be annotated `tl.constexpr`

```python
# BAD: BLOCK_SIZE is a runtime int; tl.arange needs a constexpr bound
def kernel(x_ptr, N, block_size):
    offs = tl.arange(0, block_size)               # ERROR: bound not constexpr

# GOOD
def kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)               # OK: statically known
```

## 4. `tl.dot`: the CONTRACTION dimension must be >= 16

Triton's rule is asymmetric — it reports it as
`Input shapes should have M >= 1, N >= 1 and K >= 16`. Only the shared K of
`(M,K) x (K,N)` has a floor; M and N may be smaller (an N tile of 8 compiles and runs).

```python
# BAD: BLOCK_K = 8 -> tl.dot rejects it (the contraction dim must be >= 16)
acc += tl.dot(a, b)   # a: (BLOCK_M, 8), b: (8, BLOCK_N)   ERROR

# GOOD: keep the contraction dim a multiple of 16 (16, 32, 64, ...)
BLOCK_K: tl.constexpr = 32
acc += tl.dot(a, b)   # a: (BLOCK_M, 32), b: (32, BLOCK_N)  OK
```

Multiples of 16 are the right default for **all** three dims — they map onto the tensor
core MMA shapes, so off-multiple M/N tiles are padded and waste throughput even when
they compile. But when you must pick a small value, know which dim you are shrinking:
below 16 on K is an error, below 16 on M or N is merely a slow choice.

## 5. `tl.dot`: the ACCUMULATOR must be fp32 (the input path may be tf32)

Separate two things that are easy to confuse:
- The **accumulator** dtype — must stay `tl.float32` across the whole reduction.
  A low-precision accumulator over a long reduction is what drifts past tolerance
  and fails correctness.
- The **input precision** of the multiply (`input_precision=` on `tl.dot`) — this
  is a legitimate *performance* knob, not a correctness bug. `"tf32"` runs on the
  tensor cores (~2x faster on matmul/conv), `"ieee"` runs the exact fp32 path. The
  harness's dual-precision gate accepts a tf32-matching result, so tf32 inputs are
  allowed and usually preferred for matmul/conv-bound kernels.

```python
# BAD: fp16/bf16 (or default-tf32) ACCUMULATOR -> mantissa truncation across the
#      reduction, long sums drift past the 1e-4 tolerance and fail correctness
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)   # wrong accumulator dtype
for k in range(...):
    acc += tl.dot(a, b)

# GOOD: fp32 accumulator; choose the input path on its performance merits
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # accumulator stays fp32
for k in range(...):
    acc += tl.dot(a, b, input_precision="tf32")        # tensor cores; "ieee" for exact fp32
```


## 6. `next_power_of_2` is a HOST helper — never call it in device code

```python
# BAD: tl.next_power_of_2 called inside the @triton.jit body -> JIT compile error
@triton.jit
def kernel(..., D: tl.constexpr):
    d = tl.arange(0, tl.next_power_of_2(D))       # ERROR: not valid device code

# GOOD: compute the padded size on the host, pass it in as a tl.constexpr
D_PADDED = triton.next_power_of_2(D)              # host side (python)
kernel[grid](..., D=D, D_PADDED=D_PADDED)
@triton.jit
def kernel(..., D: tl.constexpr, D_PADDED: tl.constexpr):
    d = tl.arange(0, D_PADDED)                    # OK: constexpr power-of-2 bound
    mask = d < D
```

---

Not covered here (deliberately): performance-tuning choices such as `num_stages`,
`num_warps`, block-pointer vs pointer arithmetic, or algorithm-level structure.
Those depend on the specific kernel and hardware; choose them on their merits, not
from a fixed rule. This file is only about staying correct and compilable.
