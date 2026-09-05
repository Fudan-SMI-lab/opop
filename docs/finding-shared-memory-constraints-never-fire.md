# Finding: 243 shared-memory OOMs, and the guard prevented exactly 0 of them

Found at 13:35 while reading `cand-aa016dfe`, the round-2 rewrite of L3:43's 11.0 ms family.
13 of its 40 trials died with `OutOfResources: shared memory` — and its space *does* declare
shared-memory constraints. Checking why they did not fire turned up a defect that is not
specific to this candidate, this run, or this task.

## The measurement

Every shared-memory OOM trial in the record, re-evaluated against the constraint set its own
space declared, with the real device limits substituted. Snapshot at 13:45 — the L3:43 run is
still producing trials, so these grow:

| | |
|---|---|
| shared-memory OOM trials, all runs | **251** |
| …that **violated** a declared constraint (guard should have blocked) | **0** |
| …that **satisfied every** declared constraint | **251** |

So the guard is not broken and the tuner is not sampling illegal points. The constraints
themselves do not describe the thing that fails. Every one of those 251 trials was legal by
the space's own rules, was dispatched to the GPU, compiled, and died at launch.

Distribution across runs:

```
l3-43-20260905-091705   216
l3-43-20260902-140823    26
l3-21-20260905-071312     9
```

251 of ~700 `runtime_error` trials — roughly 35% of all runtime failures, ~5% of all trials —
and about **45 minutes of GPU wall** at ~10 s each. The per-candidate rate in the current
L3:43 run is not marginal:

| candidate | trials | shared OOM | OOM % |
|---|---|---|---|
| `cand-cb7be6b4` | 80 | 34 | 42% |
| `cand-2d0194cc` | 80 | 31 | 39% |
| `cand-919059a0` | 40 | 20 | **50%** |
| `cand-de802450` | 80 | 26 | 32% |
| `cand-aa016dfe` | 40 | 13 | 32% |
| … | | | |
| `cand-3bf724d6` | 80 | 5 | 6% |

Half of `cand-919059a0`'s entire 40-trial budget went to configurations that could not launch.

## Two independent causes, both in the parameterizer's constraint set

### Cause 1: multi-kernel candidates get constraints for some kernels only

`cand-aa016dfe` launches three kernels — `_wide_row_qkv_kernel`, `_flash_kernel`,
`_linear_kernel` — each with its own tile knobs and its own `num_stages`. Its space declares
two shared-memory constraints:

```
QKV_NUM_STAGES * (QKV_BLOCK_M * QKV_BLOCK_K + QKV_BLOCK_K * QKV_BLOCK_N) * 4 <= MAX_SHARED_BYTES_OPTIN
OUT_NUM_STAGES * (OUT_BLOCK_M * OUT_BLOCK_K + OUT_BLOCK_K * OUT_BLOCK_N) * 4 <= MAX_SHARED_BYTES_OPTIN
```

QKV and OUT are covered. `ATTN_*` is not — and **all 13 OOMs are in the flash kernel**,
every one at `ATTN_NUM_STAGES` 4 or 5:

```
ATTN_NUM_STAGES=2   5 complete, 0 OOM
ATTN_NUM_STAGES=3   2 complete, 0 OOM
ATTN_NUM_STAGES=4   8 complete, 12 OOM
ATTN_NUM_STAGES=5   4 complete,  1 OOM
```

Counting kernel groups by "knob prefixes that have their own `NUM_STAGES`", and asking which
groups appear in at least one shared-memory constraint:

| | |
|---|---|
| multi-kernel spaces published | **32** |
| …with ≥1 kernel group having **no** shared-memory bound | **22 (69%)** |
| …with **zero** shared-memory constraints at all | 16 |

Examples, from `scripts/audit_shared_memory_constraints.py`:

```
l3-43-20260905-091705  sp-99ea5e68  cand-aa016dfe  groups=ATTN,OUT,QKV  MISSING=ATTN         n_shared=2
l3-43-20260905-091705  sp-d0921062  cand-919059a0  groups=ATTN,OUT,QKV  MISSING=ATTN,OUT,QKV n_shared=0
l3-43-20260905-091705  sp-55d6cb75  cand-2cda23e2  groups=PV,QK,SOFTMAX MISSING=QK,SOFTMAX   n_shared=1
l3-21-20260904-013056  sp-f95382db  cand-82819823  groups=EXPAND,FUSED  MISSING=EXPAND,FUSED n_shared=0
```

**`sp-99ea5e68` is a version-1 space, not a post-expansion one.** That matters: it means this
is *not* the constraint-dropping bug in `finding-k-expansion-drops-constraints.md`. That one is
about K discarding constraints a previous space had; this is about the constraints never being
written in the first place. Both are live, and they compound — an under-covered v1 space that
then loses what little it had.

### Cause 2: the byte width is hard-coded to 4, so the constraint is wrong at both ends

Every shared-memory constraint in the record multiplies by a literal `4`. But `COMPUTE_DTYPE`
is a knob with `["fp16", "bf16", "tf32", "ieee"]`, and the parameterizer prompt explicitly
instructs the agent to expose it (`modules.py:334-344`). fp16/bf16 stage **2** bytes per
element, tf32/ieee stage **4**.

So a single `* 4` constraint is wrong in both directions:

- **Too loose at 4 bytes** — no, it is correct there, but it is applied to *some* kernels only
  (cause 1), which is how the tf32 and ieee OOMs get through.
- **Too tight at 2 bytes** — it forbids fp16/bf16 configurations that would fit. Enumerating
  the grid of the 11 affected spaces and comparing `* 4` against `* 2`:

```
TOTAL: 7388 of 64792 2-byte grid points excluded by the hard-coded *4 (11%)
```

11% of the fp16/bf16 region is unreachable across those spaces, up to 14% in one. This is the
same dtype-byte-width confusion already recorded from the other side in
`finding-k-expansion-drops-constraints.md` — where `STATS_BLOCK_M=256` needed 139,264 B at tf32,
fit in 98,304 B at fp16, and held the best result. There the lesson was "already failed is not
infeasible when the failure was at one dtype". Here it is the same fact costing trials
systematically rather than once.

Both errors have one root: the constraint is written as if precision were fixed, when precision
is a knob in the same space.

## Why the parameterizer writes them this way

The prompt asks for constraints in three lines (`modules.py:348-352`) and gives exactly one
example:

```
Example: "BLOCK_M * BLOCK_K * 4 + BLOCK_K * BLOCK_N * 4 <= MAX_SHARED_BYTES".
```

Single kernel, unprefixed knob names, hard-coded `4`. The agent is reproducing the example
faithfully — including on candidates with three kernels and a precision knob, where the example
does not apply. The dtype instruction (step 1) and the constraint instruction (step 3) are 14
lines apart and never reference each other.

Note what the prompt does already say, and say well: the grammar section
(`modules.py:353-363`) teaches the disjunction form needed to express a dtype-dependent bound,
with a worked example that is almost exactly the fix:

```
`(DTYPE != "fp16" and 2 * BLOCK_M <= X) or (DTYPE == "fp16" and 4 * BLOCK_M <= X)`
```

So the capability is documented and the guard supports it — verified against the real
`eval_constraint`, which evaluates the four-way disjunction correctly and admits a shape at
fp16/bf16 while rejecting the same shape at tf32/ieee. What is missing is the instruction to
*use* it for shared memory, per kernel.

## What I changed — prompt only, and it is applied

`_render_prompt`'s step 3 in `agents/modules.py`. Three additions, no code semantics:

1. **Per-kernel coverage** (a): if the file launches several kernels, each needs its own
   shared-memory and thread bounds over its own knobs, and an unbounded kernel is named as the
   most common cause of wasted trials.
2. **Byte width follows the precision knob** (b): fp16/bf16 stage 2 bytes, tf32/ieee stage 4;
   a hard-coded `* 4` is wrong in both directions; write a disjunction over the precision knob.
   Also states that the fp32 accumulator stays 4 bytes regardless — only staged operands change
   width, which is the part that is easy to get backwards.
3. **Derive the arithmetic from this kernel's body** (c): the existing example is a two-operand
   fp32 matmul; a flash-attention kernel stages K and V and holds an accumulator across the
   loop. Read the `tl.load`/`tl.dot` shapes rather than copying the example.

Plus a template showing the *shape* of a per-kernel dtype-aware bound, with `<K>` for the knob
prefix and `<elements staged per stage>` left as the agent's job, and a note to use
`MAX_SHARED_BYTES_OPTIN` rather than `..._STATIC`. Verified: the prompt renders (5,563 chars),
the untouched expansion path still renders, the instantiated template passes the real
`eval_constraint` and correctly admits a 4-stage 64-column tile at fp16/bf16 while rejecting the
same shape at tf32/ieee. 191 tests pass, 9 skipped.

Point 3 is the one I am least able to shortcut, and I want to record why. My first attempt at
writing the flash kernel's bound by hand —
`ATTN_NUM_STAGES * ATTN_BLOCK_N * (ATTN_BLOCK_M + 128) * 2` — returns `True` for
`ATTN_BLOCK_M=128, ATTN_BLOCK_N=16, ATTN_NUM_STAGES=4` at tf32, which is one of the
configurations that actually OOM'd needing 114,688 B. I cannot supply the formula from outside
the kernel; only the agent reading the kernel can. So the prompt asks for a derivation rather
than offering a template to copy, and the honest expectation is that this **reduces** the OOM
rate rather than eliminating it.

## Why this is safe to change mid-run

- **Driver-side, so it affects the NEXT run only** (`opop-v2-worker-vs-driver-fix-propagation`).
  The L3:43 run in flight keeps the behaviour it started with, which also means the 243-trial
  baseline stays clean for comparison.
- **No semantics touched.** No change to the guard, the gate, the tuner, K's expansion logic,
  the convergence policy, or anything in the nine items awaiting the user's decision. It is text
  in a prompt.
- **The feasibility gate has room.** Tighter constraints shrink the feasible fraction, and
  `validation.py:167` rejects a space below 25%. Current published spaces: **min 44%, p10 66%,
  median 87%; zero below 40%.** A correctly-tightened space has substantial headroom before it
  trips the gate. This is the one real risk and it is quantified.
- **Failing the other way is visible.** If the new prompt produces over-tight constraints, the
  symptom is `infeasible_space` rejections in `SPACE_REJECTED`, which are journalled and easy to
  count. There have been none from this cause so far, so any appearance is attributable.

## What it does not fix

- The **excluded 2-byte region** (7,388 grid points) only reopens if the agent writes the
  disjunction rather than switching the literal `4` to `2`. A blanket `2` would be wrong in the
  other direction and would reintroduce the OOMs at tf32/ieee.
- **Register and thread bounds** have the same per-kernel coverage question and I have not
  measured them. The 243 figure is shared memory only; `n_regs` pressure shows up as spills and
  slow kernels rather than a clean failure, which is harder to attribute.
- **K's expansion path** (`_render_expand_prompt`) has its own constraint instructions and its
  own documented defect (26 of 26 expansions drop constraints). I am not touching it — that is
  pending item 9.

`scripts/audit_shared_memory_constraints.py` reproduces every number in this doc.
