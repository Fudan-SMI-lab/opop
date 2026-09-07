"""Bottleneck classification: which resource actually limits a kernel, without counters.

WHY THIS IS COUNTER-FREE. `ncu` is installed on the experiment boxes and FAILS:

    ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
              Performance Counters on the target device 0.

Enabling counters needs a host-side kernel-module parameter (NVreg_RestrictProfilingToAdminUsers)
that cannot be set from inside a container, and rented containers are where these experiments run.
A classifier built on `ncu` metrics would work on a workstation and return nothing on every real
box -- the same failure mode as the Triton-only profiler, one layer up. So every discriminator
below uses only CUDA events, a CPU-issue loop, torch.profiler (CUPTI), cuobjdump, and FLOP/byte
counts derived from the reference's shapes.

WHAT IT CANNOT SEE, stated so it is not mistaken for a complete account: bank conflicts, warp
divergence, instruction-cache pressure, L2 hit rate. Those need counters. `resource_limited` and
`latency_bound` are where they will be misattributed, and `latency_bound` is defined by
exclusion -- the honest reading of that verdict is "no measured resource is the limit", NOT "we
know it is latency".

ADVISORY ONLY. The verdict goes into the analyst's inputs and the bottleneck report. It never
gates, filters, or changes what runs -- same rule as every other agent suggestion here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Thresholds. Provisional pending scripts/probes/probe_bottleneck_signals.py, which measures the
# four-workload spread on the target box; the STRUCTURE is what is fixed here, not these numbers.
# Each is a fraction of a MEASURED ceiling, never a spec-sheet figure: a container's clocks are
# often capped, and a spec-derived roofline ridge misclassifies everything near the boundary.
LAUNCH_BOUND_CPU_RATIO = 1.0        # cpu_issue_ms / gpu_ms at or above this => CPU cannot keep up
SATURATED_FRAC = 0.60               # >= 60% of the measured DRAM ceiling counts as saturated
COMPUTE_SATURATED_FRAC = 0.50       # >= 50% of the measured fp32 ceiling counts as saturated
IDLE_FRAC = 0.25                    # < 25% of both ceilings means nothing is saturated
RESOURCE_NEAR_LIMIT_FRAC = 0.80     # regs/shared at >= 80% of the device limit


class DevicePeaks(BaseModel):
    """Ceilings MEASURED on the box, not taken from a datasheet."""

    model_config = ConfigDict(frozen=True)

    dram_tbs: float
    fp32_tflops: float

    @property
    def ridge_flop_per_byte(self) -> float:
        """Roofline ridge: below it a kernel is memory-side, above it compute-side."""
        if self.dram_tbs <= 0:
            return 0.0
        return self.fp32_tflops / self.dram_tbs


class BottleneckVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    # Every number behind the verdict, so the analyst can disagree with it. A classification the
    # agent cannot audit is worse than none: it would be trusted exactly when it is wrong.
    evidence: dict = Field(default_factory=dict)
    # What the class implies is worth trying. Advice, not instruction.
    suggests: str = ""


def classify(
    gpu_ms: float,
    cpu_issue_ms: float | None,
    flop_count: int | None,
    byte_count: int | None,
    peaks: DevicePeaks | None,
    n_regs: int | None = None,
    n_spills: int | None = None,
    shared_bytes: int | None = None,
    max_regs_per_thread: int | None = None,
    max_shared_bytes: int | None = None,
    empty_launch_floor_ms: float | None = None,
) -> BottleneckVerdict:
    """Classify what limits this kernel. Returns kind="unknown" when evidence is missing.

    Order matters, and it is the order of REMEDY, not of magnitude:

    1. overhead_floor  -- if GPU time is at the empty-launch floor there is nothing to optimize
                          on this structure at all, so it must be checked before anything else.
    2. launch_bound    -- if the CPU cannot issue fast enough, the GPU-side numbers describe a
                          kernel nobody is waiting for. Checked before the throughput tests
                          because those would report "nothing saturated" and send the agent to
                          look for parallelism it does not need.
    3. saturated       -- memory or compute, whichever ceiling the kernel is actually against.
    4. resource_limited-- registers/shared cap residency, so throughput CANNOT rise until the
                          tile shrinks. Checked after saturation: a saturated kernel with high
                          register use is not register-limited, it is done.
    5. latency_bound   -- the residual. Nothing measured is the limit.
    """
    ev: dict = {"gpu_ms": gpu_ms}

    if gpu_ms <= 0:
        return BottleneckVerdict(kind="unknown", evidence=ev,
                                 suggests="no valid timing; nothing can be concluded")

    if empty_launch_floor_ms and gpu_ms <= empty_launch_floor_ms * 1.15:
        ev["empty_launch_floor_ms"] = empty_launch_floor_ms
        return BottleneckVerdict(
            kind="overhead_floor", evidence=ev,
            suggests="GPU time is at this GPU's empty-launch floor: the kernel body is no "
                     "longer what costs. No tiling or precision change can help; a structural "
                     "change that removes the launch entirely is the only lever.")

    if cpu_issue_ms is not None and cpu_issue_ms > 0:
        ratio = cpu_issue_ms / gpu_ms
        ev.update({"cpu_issue_ms": cpu_issue_ms, "cpu_over_gpu": round(ratio, 3)})
        if ratio >= LAUNCH_BOUND_CPU_RATIO:
            return BottleneckVerdict(
                kind="launch_bound", evidence=ev,
                suggests="the CPU spends longer ISSUING this call than the GPU spends running "
                         "it, so halving GPU time would change nothing a caller observes. Fuse "
                         "to remove launches, and cut per-call host work (fewer ops, no "
                         "per-call allocation). Note the harness's own timing hides most of "
                         "this cost, so the real-world gain from fusing exceeds the measured "
                         "gain.")

    ai = None
    if flop_count and byte_count:
        ai = flop_count / byte_count
        ev["arithmetic_intensity"] = round(ai, 3)
    if peaks and byte_count:
        achieved_tbs = byte_count / (gpu_ms * 1e-3) / 1e12
        frac_bw = achieved_tbs / peaks.dram_tbs if peaks.dram_tbs > 0 else 0.0
        ev.update({"achieved_tbs": round(achieved_tbs, 4),
                   "pct_of_dram_peak": round(frac_bw * 100, 1)})
    else:
        frac_bw = 0.0
    if peaks and flop_count:
        achieved_fl = flop_count / (gpu_ms * 1e-3) / 1e12
        frac_fl = achieved_fl / peaks.fp32_tflops if peaks.fp32_tflops > 0 else 0.0
        ev.update({"achieved_tflops": round(achieved_fl, 3),
                   "pct_of_fp32_peak": round(frac_fl * 100, 1)})
    else:
        frac_fl = 0.0
    if peaks:
        ev["ridge_flop_per_byte"] = round(peaks.ridge_flop_per_byte, 2)

    if frac_bw >= SATURATED_FRAC:
        return BottleneckVerdict(
            kind="memory_bound", evidence=ev,
            suggests="the kernel is moving bytes at most of this GPU's measured DRAM ceiling, "
                     "so arithmetic changes cannot help. Increase reuse (larger tiles, better "
                     "blocking), or fuse to avoid writing and re-reading an intermediate.")
    if frac_fl >= COMPUTE_SATURATED_FRAC:
        return BottleneckVerdict(
            kind="compute_bound", evidence=ev,
            suggests="the kernel is near this GPU's measured fp32 ceiling. The levers are "
                     "arithmetic: tensor cores / lower precision if the accuracy gate allows, "
                     "and more independent accumulators for ILP.")

    near_limit = []
    if n_spills:
        near_limit.append(f"spills={n_spills}")
    if n_regs and max_regs_per_thread and n_regs >= max_regs_per_thread * RESOURCE_NEAR_LIMIT_FRAC:
        near_limit.append(f"regs={n_regs}/{max_regs_per_thread}")
    if (shared_bytes and max_shared_bytes
            and shared_bytes >= max_shared_bytes * RESOURCE_NEAR_LIMIT_FRAC):
        near_limit.append(f"shared={shared_bytes}/{max_shared_bytes}")
    if near_limit and frac_bw < SATURATED_FRAC and frac_fl < COMPUTE_SATURATED_FRAC:
        ev["at_limit"] = near_limit
        return BottleneckVerdict(
            kind="resource_limited", evidence=ev,
            suggests=f"neither throughput ceiling is reached, but {', '.join(near_limit)} caps "
                     "how many warps can be resident, so throughput cannot rise until the "
                     "per-thread footprint shrinks. Smaller tiles, fewer live accumulators, or "
                     "different staging.")

    if frac_bw < IDLE_FRAC and frac_fl < IDLE_FRAC:
        return BottleneckVerdict(
            kind="latency_bound", evidence=ev,
            suggests="no measured resource is the limit: both throughputs are low and nothing "
                     "is at a hardware cap, which usually means too little parallelism to hide "
                     "latency, or a serial dependency. Try more programs, split-K, or deeper "
                     "pipelining. NOTE this is the residual class -- it means 'nothing we can "
                     "measure is saturated', not 'latency is proven to be the cause'. Bank "
                     "conflicts, warp divergence and L2 behaviour are invisible here (hardware "
                     "counters are unavailable on this box).")

    return BottleneckVerdict(
        kind="mixed", evidence=ev,
        suggests="partially saturated on at least one ceiling without reaching the saturation "
                 "threshold on either; the cheapest next step is usually whichever fraction "
                 "above is largest.")
