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

## 4. `tl.dot` input dimensions must be divisible by 16

```python
# BAD: BLOCK_K = 8 -> tl.dot rejects it (needs both dims % 16 == 0, and >= 16)
acc += tl.dot(a, b)   # a: (BLOCK_M, 8), b: (8, BLOCK_N)   ERROR

# GOOD: choose tile dims that are multiples of 16 (16, 32, 64, ...)
BLOCK_K: tl.constexpr = 32
acc += tl.dot(a, b)   # a: (BLOCK_M, 32), b: (32, BLOCK_N)  OK
```

## 5. `tl.dot`: cast inputs explicitly, accumulate in fp32

```python
# BAD: relying on tf32 accumulation for an fp32 reference -> mantissa truncation,
#      long reductions drift past the 1e-4 tolerance and fail correctness
acc = tl.dot(a, b)                                # implicit tf32 path

# GOOD: keep the fp32 accumulator; use ieee precision when the reference is fp32
acc = tl.dot(a, b, input_precision="ieee")        # or "tf32" only if it still passes
# accumulator dtype stays tl.float32 across the reduction loop
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
