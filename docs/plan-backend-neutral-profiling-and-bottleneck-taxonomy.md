# Plan: backend-neutral profiling and a bottleneck taxonomy

Approved 2026-09-07. Implements steps 1–3 (profiling) and 4–5 (backend as a first-class axis),
deferring the task-classification/advisory-ranking search mechanism and CUTLASS/CuTe enablement.

## Why this exists, in one paragraph

The harness's bottleneck analysis is Triton-only and GPU-execution-only. `ProfileRecord` is
populated by reaching into Triton's compiled-kernel object, so a `cuda` candidate gets nothing —
which makes the paper's own feedback loop (tuning evidence → bottleneck report → structural
rewrite) degrade on the backend with the *higher* expressiveness ceiling. And every field it
carries describes GPU execution, so on an overhead-bound task the dominant cost is invisible to
the analyst: measured on this 4090, level2:37's reference spends **66.6 µs of CPU time issuing
launches** while the harness reports **37.9 µs total**.

## The constraint that shapes everything: no hardware counters

`ncu` is installed on the AutoDL box and **fails**:

```
$ ncu --metrics dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed python probe.py
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
          Performance Counters on the target device 0.
```

Enabling counters needs a host-side kernel-module parameter (`NVreg_RestrictProfilingToAdminUsers`)
that cannot be set from inside a container. Rented containers are where these experiments run, so:

**every classifier below must work without hardware counters.** Available signal sources:

| source | gives | works here |
|---|---|---|
| CUDA events | GPU execution time per call | ✅ (already used) |
| wall clock + CPU-only issue loop | launch/CPU-side cost | ✅ |
| `torch.profiler` (CUPTI) | per-kernel GPU time, launch counts | ✅ (counters not required) |
| `cuobjdump -res-usage` | regs / spills / shared, per kernel in a cubin | ✅ present |
| `cudaFuncGetAttributes` | same, at runtime | ✅ |
| analytic FLOP/byte from the reference's shapes | arithmetic intensity | ✅ (per-task metadata) |
| `ncu` counters | achieved occupancy, stall reasons, real DRAM bytes | ❌ **denied** |

A design that assumed `ncu` would work on a workstation and silently produce empty
`ProfileRecord`s on every rented box — the same failure mode as the Triton-only profiler, one
layer up.

## Step 1: measure the signals, and verify they discriminate

`scripts/probes/probe_bottleneck_signals.py` (written; **runs after the L1 smoke releases the
GPU** — timing anything while a run owns the card corrupts both, and the harness's lock is
per-run so nothing prevents it).

It measures the two ceilings on the box rather than trusting spec sheets (a container's clocks
are often capped, and a spec-derived ridge point misclassifies everything near the boundary),
then runs four workloads whose ground truth is known analytically:

```
COMPUTE : 4096^3 fp32 matmul       137 GFLOP / 201 MB   -> ~50 FLOP/byte
MEMORY  : 256 MB elementwise add     0 GFLOP / 512 MB   -> ~0  FLOP/byte
LAUNCH  : 40 tiny elementwise ops  trivial work, 40 launches
MIXED   : level2:37 reference       6 ops at 128x512x1024
```

**The output that decides the design is the discrimination check**: for each candidate signal, the
spread across those four workloads. A signal returning the same value for all four cannot support
the taxonomy however cheap it is, and must not be collected. This is the step that prevents
building a profiler around fields that turn out not to separate anything.

## The taxonomy: six classes, each with a counter-free discriminator

Thresholds are provisional until step 1 reports; the *structure* is what is being fixed here.

| class | limited by | discriminator | what the agent should do about it |
|---|---|---|---|
| **launch_bound** | CPU cannot issue launches fast enough | `cpu_issue_ms / gpu_ms >= ~1` | fuse to remove launches; cut per-call CPU overhead |
| **memory_bound** | DRAM throughput | AI below the measured ridge AND achieved BW >= ~60% of measured peak | improve reuse/tiling; fuse away round-trips |
| **compute_bound** | FLOP throughput | AI above the ridge AND achieved FLOP/s >= ~50% of measured peak | tensor cores, precision, more ILP |
| **resource_limited** | registers/shared cap resident warps | regs or shared near device limits, or `n_spills > 0`, while both throughputs are low | shrink tiles, cut register pressure, restage |
| **latency_bound** | too little parallelism to hide latency; serial dependencies | both throughputs < ~25% of peak AND resources idle AND not launch_bound | split-K, more programs, deeper pipelining |
| **overhead_floor** | fixed per-call cost exceeds the real work | GPU time ≈ empty-kernel launch floor | nothing left on this structure — signal `stop` |

Coverage claim, stated honestly: these six cover the bottleneck *the harness can act on*. They do
not cover things we cannot measure without counters — bank conflicts, warp divergence, instruction
cache pressure, or L2 hit rate. Those remain invisible, and `resource_limited` / `latency_bound`
are where they will be misattributed. **That limitation belongs in the report, not hidden.**

`latency_bound` is deliberately defined by exclusion (everything low, nothing saturated). It is
the residual class, and the honest reading of a `latency_bound` verdict is "no measured resource
is the limit", not "we know it is latency".

## Step 2: backend-neutral profiling via the cubin

One implementation covers three backends, because CUDA, CUTLASS and CuTe all compile through nvcc
to a cubin. `cuobjdump -res-usage` reads per-kernel registers/spills/shared out of it.

```
$ cuobjdump -res-usage kernel.cubin
Function _Z6kernelPfS_i:
  REG:42 STACK:0 SHARED:16384 LOCAL:0 CONSTANT[0]:380 TEXTURE:0 SURFACE:0 SAMPLER:0
```

Two design points learned before writing it:

1. **Aggregate over LAUNCHED kernels only, not over the cubin.** CUTLASS instantiates many
   template variants; a cubin may hold dozens of kernels of which one runs. `kernel_names`
   already carries the launched set (from the `KERNELS_NEVER_LAUNCHED` work), so intersect
   against it. Max-over-cubin would report a variant that never executed — the same class of
   error as timing a fallback path.
2. **`load_inline` needs nvcc AND ninja on PATH.** Discovered on box 2: ninja was pip-installed
   into the venv but not on PATH, so *every* CUDA candidate would have failed to build while
   `doctor` reported all green. Fixed in `7e1a2ea` (doctor now reports whether the box can build
   the cuda backend at all).

## Step 3: classification into the record, plus `cpu_issue_ms`

New `ProfileRecord` fields, all counter-free:

```
cpu_issue_ms        CPU-only time to issue the call (no GPU wait)
achieved_tbs        byte_count / gpu_time
achieved_tflops     flop_count / gpu_time
pct_of_dram_peak    against the box's MEASURED ceiling
pct_of_fp32_peak    against the box's MEASURED ceiling
bottleneck          one of the six classes
bottleneck_evidence the numbers behind the verdict, so the agent can disagree with it
per_kernel_ms       from torch.profiler, so a multi-kernel candidate can be attributed
```

Per-task metadata (computed once, not per trial): `flop_count`, `byte_count`.
Per-box metadata (measured once at `doctor` time): `device_peak_bw`, `device_peak_fp32`.

`cpu_issue_ms` is where this subsumes the launch-overhead work. **Cost containment**: measure it
in `full_eval` and baselines only, never in the 20-sample tuning trials. Tuning trials are 53.8%
of a run's wall clock and baselines 0.8%, and trial-to-trial launch cost barely varies — so the
signal costs ~0% of the run while the naive "measure everywhere" version cost ~2x. That was an
error in my first proposal, corrected here.

**The verdict is advisory.** It goes into the analyst's inputs and `BottleneckReport`; it never
gates, filters, or changes what runs. Same rule as every other agent suggestion in this harness.

## Step 4: the backend enters `structural_signature`

Today `structural_signature` is an AST hash with PARAMS zeroed — the backend is not in it, so two
candidates differing *only* in backend are not recognized as structurally distinct, and Loop D
would reject "the same idea in CUDA" as a duplicate. Small fix, real consequence.

## Step 5: one hand-seeded CUDA candidate on L2:37

Before building any search machinery for backends, get the number: **is the CUDA advantage real
under OUR timing convention?** The external team's kernel measures 4.1–4.2x over its own ATen
fallback, but 1.65x over our reference under our convention. If a CUDA candidate on our harness
is 1.05x, the case for backend search weakens sharply; if it is 2x, it is settled. One run.

## Deliberately NOT in this plan

- **Task-classification + advisory backend ranking + dynamic exploration** (the real feature).
  Two things it must get right, from evidence already on disk:
  - *Never let the ranking become a filter.* On L2:37 the seed-phase ranking was actively
    misleading: the **slowest** seed (20.256 µs) belonged to the family that improved **most**
    (−30.5%), while the **fastest** seed (14.112 µs) was the only family whose rewrites all
    failed. A ranking used as a prune would have deleted the winner.
  - *The dynamic-encouragement term needs a real denominator.* With 1–2 candidates per backend,
    per-backend success rates are noise. Untried backends must be tried **because** they are
    untried, not because of an estimated rate; weighting by measured results starts only once
    each backend has several candidates. (I have previously drawn a wrong conclusion from a ratio
    without counting its denominator — 23 expansions, 15 improved — so this is a recorded lesson,
    not a hypothetical.)
- **CUTLASS/CuTe enablement.** The profiler covers them for free (step 2), but generating them
  needs prompts, a pitfalls document, and a compile-error taxonomy per backend. Best validated on
  an H100-class card, where CUTLASS's advantage actually exists: on Ada, as the external team's
  own source notes, "dense TF32 tensor rate equals its FP32 SIMT rate".
- **Changing the timing method.** Wall-clock is added as a reported convention and an analyst
  signal. It must never become the tuning objective: it is noisier (CPU scheduling enters the
  measurement), and the search objective stays median-of-events.

---

## Progress log

### 2026-09-07 — step 2 and step 4 done, step 3 structured, step 1 still blocked

**Step 2 (backend-neutral profiling), commit `6efa850`.** `_parse_res_usage` +
`_launched_kernel_names` + `_extract_cubin_metadata` in `worker_main.py`; `profilerx.py`
consumes whichever source is present; `ProfileRecord` gained `profile_source` and
`launched_filter`. One reader (`cuobjdump -res-usage`) covers CUDA, CUTLASS and CuTe because all
three compile through nvcc to a cubin. The test exercises the real parser on genuine `cuobjdump`
output rather than a hand-written fixture.

**Step 4 (backend in the structural identity), commit `1a2d450`.** `structural_signature` now
prefixes the backend (`cuda:9f3a…`). Optional argument, so the 20 runs of recorded signatures
replay unchanged. This was the user's counterargument #3 and it was correct: without it, Loop D
would reject "the same approach expressed in CUDA" as a `duplicate_signature`, which is exactly
the exploration the backend work is meant to enable.

**Step 3 (classification) — structure written, thresholds deliberately not.**
`src/kernel_optimizer/evaluation/bottleneck.py` holds the six-class classifier with its
discriminator order, its evidence dict, and its advisory `suggests` text. The ordering is by
REMEDY, not magnitude, and each step is justified in the docstring:

1. `overhead_floor` first — if GPU time is at the empty-launch floor, nothing about the kernel
   body matters, so no later test should get to speak.
2. `launch_bound` before the throughput tests — otherwise a launch-bound kernel reports "nothing
   saturated" and the agent is sent looking for parallelism it does not need.
3. `memory_bound` / `compute_bound` — against MEASURED ceilings, never spec-sheet ones.
4. `resource_limited` after saturation — a saturated kernel with high register use is not
   register-limited, it is finished.
5. `latency_bound` last, as the residual.

The thresholds in that file are provisional and marked as such. A sanity check on analytically
known inputs already shows why they must be measured rather than chosen: a 4096³ fp32 matmul at
10 ms on a 30 TFLOPS ceiling lands at **45.8% of peak**, which falls on the wrong side of a
provisional 50% `COMPUTE_SATURATED_FRAC` and classifies as `mixed` instead of `compute_bound`.
That is one guessed constant deciding a verdict, which is the failure this step's ordering exists
to prevent.

**Step 1 is still blocked, and the blocker is the point.** `probe_bottleneck_signals.py` is
staged on box 2 but must not run while anything else owns the GPU: it measures ceilings, so a
concurrent tenant silently lowers every number it reports and would bake a too-low ceiling into
the thresholds permanently. The harness's GPU lock is per-run, so nothing prevents the collision
mechanically — it has to be sequenced by hand. A momentarily idle card during an agent call is
NOT a free card; tuning trials resume the instant the call returns.

### Unplanned: two live defects found while waiting, commit `792bc10`

Both were found on run-l1-42-20260907-193510 and neither is specific to that task.

**A read timeout is not a call ceiling.** `request_timeout_s` is httpx's per-READ idle timeout,
so it fires only on silence. One rewriter call ran **4057 s (67.6 min) against a 1500 s setting**
— 2.7× — because the agent kept emitting tool calls throughout. I had previously recorded 1500 s
as the hard ceiling on an agent call; that is wrong, and no value of `request_timeout_s` fixes
it, because the failure mode is a talkative call rather than a silent one. Added
`total_call_timeout_s` (2100 s), enforced by a watchdog that aborts the session **and closes the
transport** — measured, not assumed: abort alone returns 200 and ends the turn server-side while
the already-streaming POST never returns, leaving the client thread blocked forever.

**An agent's own script can take the machine down.** The same rewriter wrote a parameter sweep
whose kernels included one producing a **272,341-line PTX**; `ptxas` then reached **111 GiB
resident**, drove the container's memory cgroup to its limit (125.5 GB against a 124.5 GB
`memory.high`, 16.0M throttle events) and put the orchestrator in **D-state on
`mem_cgroup_handle_over_high`** — which presents as "the agent call is stuck on the socket" and is
not. Three GB from a hard OOM. This is the **second** occurrence of this root cause; the first
(1500+ Triton compiles in a self-check) lost a finished rewrite to the ceiling, and the fix
recorded then was never implemented. Now in the candidate contract's "Your own testing" section,
which binds every code-writing agent rather than the rewriter alone.

Relevance to this plan: the profiler being built here measures ceilings and per-kernel time, and
both are meaningless if a neighbouring process is competing for the GPU or the box is being
throttled. The same discipline that makes step 1 wait is what these two fixes enforce
automatically.
