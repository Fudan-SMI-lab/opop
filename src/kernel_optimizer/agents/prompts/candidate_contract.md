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
- **Four Python constructs are rejected outright by a static check, before your file
  reaches the GPU.** They are banned because each was used to fake a passing kernel,
  and the check is a plain regex that cannot tell your intent from that one:
  - **`try:` / `except`** — used to fall back to PyTorch whenever the custom kernel
    raised, so an unfinished kernel still passed. Write no exception handling; if an
    input case needs different handling, branch on it explicitly.
  - **`pass`** — used to inherit the reference class and do nothing, so the parent
    implementation did the work. Use an explicit no-op you actually need instead (e.g.
    `return x` , or restructure the branch away).
    **The check matches the WORD `pass` anywhere outside a comment, including inside a
    string literal.** Comments are stripped before matching, strings are not — so
    `_NOTE = "we pass tiles through shared memory"` fails the candidate while the same
    text after a `#` is fine. Prefer "hand off"/"single-stage" in strings. (`passed`
    and `bypass` are safe: the match is on word boundaries.)
  - **`threading` / `multiprocessing` / `concurrent.futures`** — used to manipulate
    timing. Even an unused `import threading` fails.
  These are hard rejections, not warnings: the file is refused with a message about
  the cheat the pattern is associated with, which will NOT describe what you were
  doing. Avoiding the four constructs is much cheaper than arguing with the message.
- Correctness is checked by a diff-test against the reference on its own input
  shapes. The tolerance is tight: under the strict fp32 mode the harness uses
  `torch.allclose(atol=1e-4, rtol=1e-4)` over the WHOLE output tensor (any single
  element out of tolerance fails the candidate). Under the dual-precision mode the
  harness compares against the reference computed at BOTH tf32 and ieee fp32
  precision and accepts if your output matches EITHER (relative error < 1% on
  >99% of elements, AND cosine similarity >= 0.99985 — both must hold). Either way,
  do not assume loose slack — a numerically sloppy reduction (e.g. tf32 accumulation
  over a long dimension) will be rejected.

## Your own testing: keep it small, and never sweep

The harness evaluates every file you produce — correctness against the reference on
its real input shapes, then timing over many samples, in its own clean process. You
do not need to reproduce that, and you must not try to.

A quick sanity check of ONE configuration on ONE small shape is welcome (it catches
a typo before it costs a round). Past that, every additional check costs the round
that produces your output, and buys nothing the harness is not about to measure
anyway. **Concretely, in your own scratch scripts:**

- Do NOT sweep parameter combinations to find a fast one. Parameter tuning is a
  separate automatic stage that runs after you; picking values is not your job, and
  a config that wins your 20-iteration timing is not the one it will pick.
- Do NOT loop over many shapes, kernel sizes, strides, paddings, or dtypes. If you
  test at all, test the shape in the task description.
- Do NOT compile the same kernel more than a handful of times. Each JIT compile is
  seconds of wall clock, and a nested loop reaches thousands without looking like it.
- Never build a kernel with a huge grid, a huge unroll factor, or a loop bound
  derived from a large shape just to see if it compiles. A single such kernel can
  produce a PTX file hundreds of thousands of lines long, and the assembler will
  then consume **more RAM than the machine has** trying to compile it. Measured: one
  such kernel reached 111 GiB resident and stalled every other process on the box
  for 47 minutes until it was killed. Your work was lost with it.

The hard rule behind all of it: **your call has a wall-clock ceiling, and the file
you never finished writing is worth nothing.** Write the file first. If you have
budget left afterwards, then sanity-check it.

## Reference run mode (train vs eval) — read `task/eval_semantics.md`

The reference model is evaluated in a specific run mode, and `task/eval_semantics.md`
tells you which one (the harness observed the live model). **Your kernel must
reproduce the semantics of THAT mode**, not an assumed one. This matters most for
normalization layers:

- **TRAIN mode**: `nn.BatchNorm2d`/`InstanceNorm` normalize with the **current
  batch's** mean/var (computed from the input), then apply the affine `weight`/`bias`.
  They do **NOT** use `running_mean`/`running_var` — on a freshly constructed,
  untrained model those are `0`/`1`, so using them produces a large systematic error
  (e.g. everything off by a near-constant amount). If `eval_semantics.md` says the
  reference (or a given norm layer) is in train mode, compute batch statistics.
- **EVAL mode**: normalization uses the stored `running_mean`/`running_var`.
- Dropout is a no-op when its `p == 0` regardless of mode; otherwise it too depends
  on the mode.

Match each layer's stated `training` flag; do not assume all layers agree.

## Precision and the tensor-core path

For matmul- and convolution-bound kernels, the arithmetic precision of the dot
product is often the single largest lever on latency — larger than block sizes,
warps, or stages. On this GPU, `tl.dot(..., input_precision="ieee")` runs on the
scalar FMA path and leaves the tensor cores idle; `input_precision="tf32"` (or
casting inputs to fp16/bf16 with an fp32 accumulator) dispatches to the tensor
cores. **How much faster that is on THIS box is measured, not assumed — read the
ratio off `docs/device.md`, which carries this machine's own ceilings.** `torch.compile` gets its
speed from exactly this — it uses the tf32 tensor-core path by default.

Because of that, you should treat dot-product precision as a first-class design
choice, not an afterthought:

- Under the dual-precision correctness mode (the L3 experiments use it) the
  harness accepts a result that matches the reference computed at **tf32**. A
  tf32 tensor-core kernel is therefore a legal, accepted candidate. Make the
  tensor-core path *reachable* for matmul/conv-bound work — but **do not commit to one
  precision in the source**: which precision actually wins is decided by the tuner on
  real measurements, and it varies by task. Measured here: fp16 and bf16 tied on one
  attention task (3.03 vs 3.01 ms), bf16 failed correctness outright on a state-space
  task where fp16 passed, and on a third task tf32 and fp32 differ substantially in
  throughput (the measured factor for THIS box is in `docs/device.md` — it ranges from
  under 2x to nearly 6x depending on the card, so do not assume a value).
  None of that is predictable from the source, which is why the next bullet matters more
  than any default you might pick.
- **A tile chosen for one precision often cannot launch at another, and that has cost us
  a whole precision branch.** A tile sized at 2 bytes/element (fp16/bf16) can need
  131072–164352 bytes of shared memory at 4 bytes/element (tf32/ieee) against a
  ~101376-byte limit — so the candidate simply fails to launch there, the tuner never
  measures it, and the winning configuration was never compared against that precision.
  Keep the tile domain wide enough at the low end that every precision you offer has at
  least one launchable configuration. **You do not need to compute the shared-memory
  figure** — the harness gets it from the compiler (see the note under constraints).
- Make the compute precision a SINGLE tunable knob that controls the WHOLE
  precision path — both the input cast and the `tl.dot` precision — e.g.
  `PARAMS["COMPUTE_DTYPE"] = "tf32"` with choices `["fp16", "bf16", "tf32", "ieee"]`:
  `"fp16"`/`"bf16"` cast the dot inputs to `tl.float16`/`bfloat16` (tensor cores);
  `"tf32"`/`"ieee"` keep fp32 inputs with the matching `input_precision`. Keep an
  `"ieee"` choice reachable so a candidate that genuinely needs full fp32 can still
  be expressed. **REQUIRED:** if your kernel uses a low-precision cast
  (`.to(tl.float16)` etc.) to reach the tensor cores, that dtype MUST come from a
  PARAMS knob — never hard-code fp16 in the body while leaving PARAMS without a
  dtype knob (the tuner then cannot compare precisions and the fp16 path is never
  measured against the alternatives).
- Precision applies to the *dot/accumulate* step. Even on the tf32/fp16 input path,
  **keep the accumulator in fp32** (`tl.zeros(..., dtype=tl.float32)`) unless you have a
  specific reason not to: low precision only reduces the mantissa of the multiply inputs,
  not the accumulation, and an fp32 accumulator over a long reduction is what keeps you
  inside tolerance. A sloppy low-precision *accumulator* is the thing that fails the
  diff-test, not a tf32/fp16 *input*. This is a strong default rather than a checked rule
  — nothing rejects a candidate for its accumulator dtype — so if you believe a shorter
  reduction tolerates a narrower accumulator, you may try it, but the diff-test is then
  the only thing standing between that choice and a wrong answer: verify it on the task's
  real shape, and say so in `approach_summary`.

### The precision knob must vary the ARITHMETIC only, never the algorithm

This is the single most expensive mistake measured in this harness, and it is worth stating
separately because it does not look like a mistake in the source. **Two distinct failures,
both reported identically as `correctness_mismatch`, and both are the candidate's own
doing** — the acceptance gate was audited over 537 rejections and attributed 0 of them to
itself (rejected candidates missed the fp64 reference by 3.04x at best, 6.48x median, up to
3766x, so they were not near-misses).

**(1) `PREC` gating the algorithm rather than the dot.** Where the precision variable is
confined to a small dot helper, it really is a precision knob. Where it also selects a
*code path*, the non-default branches are running code that was never stabilized, and the
knob is no longer measuring precision — it is measuring which branch you debugged.
Measured over four candidates on one task, by counting `if PREC ==` sites per candidate:

```
PREC only inside the dot helper      fp16 32/32, bf16 17/17, tf32 12/12  PASS
PREC only inside the dot helper      fp16 15/15, bf16 41/41 pass; tf32   0/11 FAIL
PREC in the dot helper AND the ALGORITHM   fp16 43/43 pass; bf16 0/13, tf32 0/9  FAIL
PREC in the dot helper AND the ALGORITHM   fp16 pass;       bf16 0/12, tf32 0/4  FAIL
```

**So: write the algorithm once. `COMPUTE_DTYPE` may change casts and
`input_precision`, and nothing else.** If a precision genuinely needs a different
numerical treatment (a log-space stabilization, a rescaling), apply that treatment
**unconditionally at every precision** — a stabilization that is correct at bf16 is not
wrong at fp16, it merely costs a little. A branch that only one precision enters is a
branch the tuner cannot compare, and the winning configuration was never measured against
the alternatives.

**(2) An uncompensated `dot` where the mantissa is genuinely too short.** A single
low-precision MMA truncates the multiply inputs' mantissa. Where that is what fails, the
fix is arithmetic, not a branch — split each operand into a high and a low part and
accumulate the cross terms, which roughly doubles the effective mantissa for about 3x the
MMAs (still far cheaper than the scalar path):

```python
ah = a.to(tl.float16); al = (a - ah.to(tl.float32)).to(tl.float16)
bh = b.to(tl.float16); bl = (b - bh.to(tl.float32)).to(tl.float16)
acc += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)   # 3 MMAs, ~2x mantissa bits
```

Measured on the task that motivated this: the split form passed **12/12** correctness
checks where the single-MMA form failed **11/11**.

**Offer the compensation as its own knob, so the tuner decides whether the task needs
it** — do not hard-wire either form:

```python
PARAMS = {
    "COMPUTE_DTYPE": "tf32",     # ["fp16", "bf16", "tf32", "ieee"]
    "DOT_MODE": "plain",         # ["plain", "split3"]  -- arithmetic only, NOT a code path
}
```

`"plain"` is one MMA; `"split3"` is the three-MMA form above. Both must produce the same
algorithm with the same shapes — the only difference is how the product is computed. On a
task whose reduction is short or whose values are small, `"plain"` wins on speed and
`"split3"` costs 3x the MMAs for nothing; on a long or wide-dynamic-range reduction the
reverse. **That is exactly the kind of question the tuner exists to answer on
measurements, and it cannot answer it if only one form is expressible.**

**A third failure mode is fp16-specific and is about RANGE, not mantissa**, so neither
knob above addresses it: fp16 overflows to Inf/NaN above 65504. If the task's intermediate
values can exceed that (check the reference's magnitude), fp16 needs rescaling or must be
left out of the knob's choices — while bf16 keeps fp32's exponent range and does not have
this problem. Diagnose which of the three you are facing before treating a low-precision
failure as a reason to abandon low precision: on one task 8 of 8 tensor-core candidates
were rejected and 7 of 7 scalar candidates accepted, and the resulting best kernel used no
tensor cores at all.

## Backend

- **Two backends are supported: `triton` (`@triton.jit` kernels) and `cuda` (via
  `torch.utils.cpp_extension.load_inline`).** Declare which one you used. CUTLASS/CuTe
  and TileLang are NOT available in this harness's worker environment — a candidate
  using them fails to compile, so do not reach for them.
  **Start from Triton** for a practical reason, not a stylistic one: the harness reads
  register counts, shared-memory usage and pipelining depth straight out of the Triton
  compiler, so a Triton candidate gets a resource profile and a bottleneck report that a
  CUDA candidate gets less of; and its compile cycle is seconds rather than a minute.
  **Choose `cuda` when Triton cannot express or cannot reach what you need**, and say so
  in `approach_summary`. Two cases measured on this hardware:
  - **Strict IEEE fp32 dot products.** `tl.dot(..., input_precision="ieee")` has no fast
    path here; a hand-written CUDA attention kernel reached 55–74% of this card's fp32
    CUDA-core roof where the best of 36 Triton tile configurations reached 18%, and no
    tile closed the gap. If the task genuinely needs full fp32 arithmetic, CUDA is the
    stronger backend.
  - **Warp-level primitives or memory instructions Triton does not expose** (explicit
    `__shfl_*`, a specific `cp.async` shape, `__launch_bounds__`). Note that Triton's
    `num_stages` already emits `cp.async` double-buffering — verified in SASS — so
    pipelining alone is not a reason to leave Triton.
- torch operations are allowed around and between your custom kernel(s) — layout,
  reshaping, and also **computation**, including the vendor libraries.
  **When to hand a sub-op to the vendor library.** A large, regular GEMM or convolution
  through `F.linear` / `F.conv2d` runs on cuBLAS/cuDNN, which is usually already near
  the hardware roof for that shape. Rewriting it yourself often just reproduces it more
  slowly; writing your kernel for what *surrounds* it can be the larger win. Measured on
  this project: at strict IEEE fp32, cuBLAS beat a hand-written Triton GEMM by 1.33x on
  the two projections of an attention block — while at tf32/fp16/bf16 the hand-written
  Triton GEMM matched cuBLAS to within 5% and reached 87–93% of the measured roof, so
  there the library buys nothing.
  **When to write it yourself instead.** The library call is a hard boundary: nothing
  fuses across it. If you can fold the surrounding elementwise work, normalization
  statistics, or a reduction *into* the matmul kernel and save a full read/write of a
  large tensor, your own kernel can beat the library even when it is slower in isolation.
  That is a real result on this project too — the winning MBConv candidate made the
  BatchNorm statistics a by-product of a kernel that was already reading the data.
  **This is your judgement to make, and you must state it**: say in
  `approach_summary` which sub-ops you handed to the library and which you kept, and why.
  **The hard floor does not move**: your file must define at least one real kernel, and
  the computation you claim to optimize must run in it. A file that only calls torch ops
  is rejected.
- **Never call `torch.compile`, `torch.jit.script` or `torch.jit.trace`.** These are
  rejected by a static check before evaluation, whatever else the file contains. The
  reason is not style: the baseline your candidate is measured against IS
  `torch.compile` on the reference module, so a candidate that compiles the same graph
  is compared with itself and any difference it shows is scheduling noise. Adding a
  small kernel next to a compiled graph does not change that — the compiled graph is
  still doing the work — so this is rejected even when a `@triton.jit` kernel is
  present. Plain eager torch ops around your kernel are fine.
- A file that defines **no** kernel at all (no `@triton.jit`, no inline CUDA
  extension) is rejected for the same reason.
