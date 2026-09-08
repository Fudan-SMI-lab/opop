# Backend coverage: what is built, what is reachable, and what is measured

Written 2026-09-08. The question this answers: it was decided earlier that different task types
may suit different backends (Triton / CUDA C++ / CUTLASS / CuTe), and that CUTLASS/CuTe would be
used. This records what actually exists in the harness today, what a run actually produced, and
three defects found by testing the non-Triton path end to end on box 2 rather than reading the code.

## The result first: every candidate ever produced on this box is Triton

Audited every run on box 2 by counting the `backend` field of `CANDIDATE_REGISTERED`:

| run | candidates | backends |
|---|---|---|
| run-l3-43-20260908-053708 | 19 | triton 19 |
| run-l1-42-20260908-023039 | 8 | triton 8 |
| run-l1-42-20260907-193510 | 6 | triton 6 |
| run-l1-42-20260908-015408 | 2 | triton 2 |
| **total** | **35** | **triton 35, cuda 0, cutlass 0, cute 0** |

Confirmed against the sources on disk, not just the declared field: all 35 contain `@triton.jit`,
none contains `load_inline`, `cpp_extension`, `cutlass`, or `cute::`. Six also call torch/cuBLAS
matmul alongside their Triton kernels (`F.linear` for the projections), which is the only non-Triton
arithmetic that has ever run — and it is a library call, not a written kernel.

So: **there is no measured evidence about backend choice at all.** No CUDA candidate, no CUTLASS
candidate, no comparison. Any claim that Triton is the right backend for these tasks is currently
an artifact of what was asked for, not a finding.

## Why: the contract tells agents to prefer Triton, and CUTLASS is not offered

`agents/prompts/candidate_contract.md`:

> - Prefer `triton` (`@triton.jit` kernels). CUDA via
>   `torch.utils.cpp_extension.load_inline` is allowed if declared.

That is the whole backend section. CUTLASS and CuTe are **never mentioned to the agent**, in any
prompt. `Backend = Literal["triton", "cuda"]` (`models/core.py:18`) does not admit them as values.
Agents did exactly what they were told; 35/35 Triton is compliance, not a preference they formed.

The *infrastructure* for other backends is real and was deliberately built backend-neutral: one
cubin reader covers CUDA C++, CUTLASS and CuTe because all three compile through nvcc
(`worker_main.py:339-344`), `families.py:90` reasons about it, and `profilerx.py` records
`profile_source` so a reader can tell a Triton measurement from a cubin one. That work is sound. It
is just never exercised.

## Three defects, found by running the CUDA path rather than reading it

Tested end to end on box 2 with a minimal `load_inline` candidate against a real reference, through
the actual `worker_main.py` eval job.

### D1. `cubin` metadata is unreachable for every non-Triton candidate

`worker_main.py:1919` gates BOTH metadata branches on one flag:

```python
if correct and job.get("collect_triton_metadata"):
    if backend == "triton":  ... result["triton"] = ...
    else:                    ... result["cubin"] = ...
```

and every caller sets that flag as `collect_triton_metadata=(backend == "triton")`
(`correctness.py:81, 100, 126, 147`). So for `backend="cuda"` the flag is False, the outer `if`
fails, and the `else` branch — the one written specifically to serve cuda/cutlass/cute — can never
execute. Measured: a correct CUDA candidate returns `cubin: None`, `cubin_error: None`. Flipping the
flag by hand on the same job immediately produced a full cubin record (18 kernels with regs /
spills / shared). The reader works; the gate never opens.

Consequence: a CUDA/CUTLASS candidate would run and be timed, but arrive at the bottleneck
classifier with no occupancy, no registers, no spills, no shared bytes — so it gets a strictly worse
verdict than a Triton candidate for reasons that have nothing to do with the kernel. The comment at
`worker_main.py:1928` says this was already fixed once ("Before this, `backend == "triton"` gated
the whole block and a cuda candidate carried no resources at all") — the inner gate was fixed and
the outer one was left, so the defect survived the fix.

**FIXED** (`_wants_kernel_metadata` in `worker_main.py`; job key renamed to
`collect_kernel_metadata`, old key still honoured for replay; the four `correctness.py` call sites
now pass `True` unconditionally). Verified on box 2 with the same probe: `cubin` present,
1 kernel, `regs=12 spills=0 shared=0`, where before the fix it was `cubin: None`.

### D2. The launched-kernel filter compares mangled names with demangled ones

`_launched_kernel_names` returns demangled signatures — measured: `addk(float const*, float
const*, float*, int)`. `_extract_cubin_metadata` reports mangled symbols — `_Z4addkPKfS0_Pfi`.
They can never match, so `launched_filter` came back `no_name_match` and the record covered **18
kernels of which 17 were never launched** (leftovers from an unrelated probe sharing the extension
cache). The union's max regs/shared then describes some other kernel entirely.

This matters most for exactly the backend that was asked for: CUTLASS instantiates many template
variants, and `worker_main.py:569` states plainly that reporting the union "would report a variant
that never executed -- the same class of error as timing the wrong thing". The guard was written
with the right intent and does not work.

**FIXED** (`_kernel_identifiers` in `worker_main.py`): both dialects are reduced to bare function
identifiers and compared as SETS, not substrings — a substring test would let a short name like `mm`
match an unrelated `mmadd2`. Itanium length-prefixed components cover nested CUTLASS names
(`_ZN7cutlass6kernel4gemmEv` yields `{cutlass, kernel, gemm}`); demangled names lose their template
arguments and argument list. Verified on box 2: `launched_filter` went from `no_name_match` with 18
kernels to `applied` with the 1 kernel that actually ran. `no_name_match` is kept and still loud —
it correctly reported its own failure, which is the only reason this was visible at all.

### D3. The worker accepts backend names the model cannot express

`worker_main.py:1607` branches on `backend.lower() in ("triton", "tilelang", "cute")` to choose the
module loader, but `Backend = Literal["triton", "cuda"]` means `"cute"` and `"tilelang"` can never
reach it through a validated `Candidate`. So the code is either dead or the type is wrong. Since
CuTe kernels are Python (`cute.jit`-decorated) and must load like Triton rather than through
`load_inline`, that branch is presumably the correct future behaviour — the type is what is behind.

**Fix:** decide the supported set once, in `Backend`, and make the loader and the contract agree
with it. Do not add a backend to the Literal until the path is tested end to end, or D1/D2 will
simply repeat for it.

## What this means for the "different tasks suit different backends" question

Nothing has been measured yet, and the harness cannot currently measure it fairly: D1 alone means a
non-Triton candidate is profiled blind, which biases the bottleneck feedback and therefore the
rewrite loop against it. Fixing D1+D2 is a precondition for any backend comparison, not an
improvement to one.

Sequence, once D1-D3 are fixed:

1. **Verify the CUDA path carries resources**, using the same probe as above but asserting a
   non-empty `cubin` and `launched_filter != "no_name_match"`.
2. **Offer CUDA as a real option in the contract** rather than a discouraged one, and give the
   generator a reason to choose it — the honest one available today is that a task whose verdict is
   `resource_limited` on shared memory may do better with a hand-managed layout than with Triton's
   pipeliner, which is a hypothesis the harness can test.
3. **Only then consider CUTLASS/CuTe.** It needs the dependency present in the worker venv (not
   currently installed — that alone makes any CUTLASS candidate a compile error today), plus a
   contract section, plus D2 fixed or its resource numbers will describe template variants that
   never ran.

Note the existing evidence bearing on this, which argues for measuring rather than assuming: a
prior audit found handwritten CUDA losing to Triton at 0.874x on the same algorithm under a
double-sided configuration sweep, and KernelPro's 1.23x came from CuTe rather than from plain CUDA C.
So the interesting comparison is Triton vs CuTe, and plain CUDA C++ is mainly worth having as the
path that makes CuTe reachable.

## Priority

Below P1 (the 18% shared-memory trial waste) and P2/P3 (launch_bound, fp16 ceiling), because those
three affect every run happening now while this affects a capability not yet in use.

**D1 and D2 are now fixed and verified**, since each was correct code behind a broken gate and both
were cheap. Each new test was checked against a negative control — the fix reverted in place, the
test required to fail — because the first attempt at that control was itself invalid: it patched a
copied tree while the editable install kept importing the real `src`, so both controls "passed" while
testing the fixed code. Patching in place, both correctly fail on the pre-fix code.

D3 (the `Backend` Literal vs the loader's `"cute"`/`"tilelang"` branch) is deliberately NOT fixed:
deciding the supported set is a scope decision, and adding a value to the Literal before its path is
tested end to end is precisely how D1 and D2 came to exist.

Note what this does and does not buy. It does not produce a single non-Triton candidate — the
contract still says "prefer triton" and CUTLASS is still unmentioned and uninstalled. It removes the
measurement bias that would have made the first such candidate look worse than it is.
