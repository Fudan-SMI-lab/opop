"""Bottleneck classification: which resource actually limits a kernel, without counters.

WHY THIS IS COUNTER-FREE. `ncu` is installed on the experiment boxes and FAILS:

    ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
              Performance Counters on the target device 0.

Enabling counters needs a host-side kernel-module parameter (NVreg_RestrictProfilingToAdminUsers)
that cannot be set from inside a container, and rented containers are where these experiments run.
A classifier built on `ncu` metrics would work on a workstation and return nothing on every real
box -- the same failure mode as the Triton-only profiler, one layer up.

WHAT IS ACTUALLY VISIBLE WITHOUT COUNTERS. An earlier version of this docstring claimed tensor
cores, occupancy, spills and shared traffic were all invisible here. That was too pessimistic and
step 5 disproved it by measurement on box 2:

  tensor cores    VISIBLE. Disassembling the cubin (nvdisasm/cuobjdump, no privileges needed)
                  recovers HMMA/IMMA/BMMA/OMMA. An fp16 tl.dot kernel shows 16; a scalar
                  elementwise kernel shows 0.
  spills          VISIBLE as STL/LDL, cross-validated against Triton's own n_spills (STL=2/LDL=1
                  against n_spills=2, agreeing exactly).
  shared traffic  VISIBLE as LDS/STS, and barriers as BAR.SYNC.
  occupancy       COMPUTABLE analytically from n_regs/shared/num_warps plus device properties,
                  including WHICH resource binds it.
  vectorization   VISIBLE as the width of LDG/STG (.128 / .64 / .32).

WHAT REMAINS INVISIBLE, stated so this is not mistaken for a complete account: shared-memory bank
conflicts, warp divergence, instruction-cache pressure, L1/L2 hit rates, stall-reason breakdown,
and ACHIEVED (as against theoretical) occupancy. `latency_bound` is where those get misattributed,
and it is defined by exclusion -- the honest reading of that verdict is "no measured resource is
the limit", NOT "we know it is latency".

EVERY THRESHOLD IS MEASURED, NOT GUESSED. The lines come from `evaluation/calibration.py`, derived
from four workloads whose bottleneck is known analytically, on the box the run is happening on.
Two constants that used to live here were disproved that way:

    COMPUTE_SATURATED_FRAC = 0.50  -- a genuinely compute-bound matmul reaches 94.9% of the
                                      measured fp32 ceiling, so a 0.50 line declares victory on a
                                      kernel with 2x of headroom left.
    LAUNCH_BOUND_CPU_RATIO = 1.0   -- an indisputably launch-bound workload measures 0.963, i.e.
                                      BELOW the line, so the test judged backwards and could
                                      never fire.

Passing no `thresholds` falls back to the derived DEFAULTS in calibration.py (which are documented
there and deliberately conservative), never to those two.

ADVISORY ONLY. The verdict goes into the analyst's inputs and the bottleneck report. It never
gates, filters, or changes what runs -- same rule as every other agent suggestion here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.evaluation.calibration import Thresholds

# Registers/shared at this fraction of the device limit count as "near the limit". Not a
# calibrated quantity: it is a fraction of a HARD architectural limit (255 registers, 100 KB of
# shared memory), not of an achievable throughput, so there is nothing to measure -- the limit is
# the same number the compiler is working against.
RESOURCE_NEAR_LIMIT_FRAC = 0.80
# Occupancy below this is worth telling the agent about. Chosen from the measured distribution on
# box 2 rather than from a round number: a plain kernel reaches 1.00, a 112-register real candidate
# 0.33, a 218-register tile 0.17. A line at 0.5 separates "the tile is costing residency" from
# "residency is fine", and is expressed as occupancy (already dimensionless) so it travels.
LOW_OCCUPANCY_FRAC = 0.50

# Fallback thresholds when no calibration is available. These are the documented defaults from
# calibration.py's `derive_thresholds`, restated here so a classifier call with `thresholds=None`
# behaves identically to one with an uncalibrated box -- and so the disproved constants cannot
# creep back in as "defaults".
DEFAULT_THRESHOLDS = Thresholds(
    dram_saturated_frac=0.60,
    compute_saturated_frac=0.80,      # HIGH on purpose: measured evidence puts a genuinely
                                      # compute-bound kernel near 0.95
    idle_frac=0.25,
    launch_bound_cpu_ratio=0.85,      # NOT 1.0, which measurement showed is above what a
                                      # launch-bound workload exhibits
    derivation={"source": "uncalibrated fallback; run `kernel-opt calibrate` to derive these "
                          "from this box's own measured separation"},
)


class DevicePeaks(BaseModel):
    """Ceilings MEASURED on the box, not taken from a datasheet.

    Kept as a small separate type (rather than passing a whole `Calibration`) so the classifier
    can be exercised with two lines of setup, and so a caller that only has ceilings -- an
    external probe, a replayed worker result -- can still use it.
    """

    model_config = ConfigDict(frozen=True)

    dram_tbs: float
    fp32_tflops: float
    # The tensor-core ceiling, when known. Load-bearing: on the 4090 it is 1.62x the fp32 figure,
    # so a tf32 kernel measured against fp32 reads as >100% of peak (nonsense that looks like
    # "done") and a scalar kernel measured against tf32 reads as hopeless. `classify` picks the
    # denominator from the kernel's OWN instruction mix when it can.
    tf32_tflops: float = 0.0

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
    # Where the two methods disagreed, and which one this verdict follows. KernelPro's
    # cross-validation practice: when a throughput-fraction reading and the analytic roofline
    # position conflict, defer to the ANALYTIC bound, because the fractions depend on a measured
    # ceiling that a busy box can depress while the roofline position depends only on the task's
    # own FLOP/byte ratio.
    disagreement: str = ""
    # Signals that are unavailable on this box, so a reader does not take their absence for a
    # clean bill of health.
    unmeasured: list[str] = Field(default_factory=list)


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
    thresholds: Thresholds | None = None,
    sass: dict | None = None,
    occupancy: dict | None = None,
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
    th = thresholds or DEFAULT_THRESHOLDS
    ev: dict = {"gpu_ms": gpu_ms, "thresholds": {
        "dram_saturated_frac": th.dram_saturated_frac,
        "compute_saturated_frac": th.compute_saturated_frac,
        "idle_frac": th.idle_frac,
        "launch_bound_cpu_ratio": th.launch_bound_cpu_ratio,
        "calibrated": thresholds is not None,
    }}
    unmeasured = _unmeasured_signals()

    # Tier 1 facts about the COMPILED KERNEL, recorded before any verdict can return.
    #
    # These describe the binary, not the bottleneck, so they are true regardless of which branch
    # below fires -- and every branch's reader wants them. They were previously written only
    # inside the `near_limit` block near the end, which four of the five verdicts return before
    # reaching (`overhead_floor`, `launch_bound`, `memory_bound`, `compute_bound`). The effect was
    # an inconsistent facts section: `uses_tensor_cores` appeared (recorded further down but still
    # before those returns) while the equally-cheap, equally-Tier-1 occupancy did not.
    #
    # Observed on run-l1-42-20260908-023039: verdict `memory_bound` at 94.7% of the measured DRAM
    # ceiling, whose advice reads "increase reuse (larger tiles, better blocking)" -- addressed to
    # a kernel whose profile held `occupancy 0.3333, limiter blocks_per_sm` and was therefore
    # already at the per-SM block cap. Tier 1 had measured it; the classifier dropped it.
    #
    # Recording the fact is deliberately NOT the same as making it a lever: occupancy stays out of
    # `near_limit` for a saturated kernel. That separation is enforced by the early returns above,
    # not by the `near_limit` gate's own fraction conditions -- swept across both fractions, no
    # input reaches that gate with either fraction above its saturation line, because
    # `memory_bound`/`compute_bound` already returned. Those conditions are therefore dead code
    # kept as documentation of intent, and removing them changes no verdict. Facts here; the
    # ranking of levers stays where it was.
    occ_frac = (occupancy or {}).get("occupancy")
    if occ_frac is not None:
        ev["occupancy"] = round(float(occ_frac), 4)
        ev["occupancy_limiter"] = (occupancy or {}).get("limiter")
    if n_spills:
        ev["n_spills"] = n_spills
    if n_regs:
        ev["n_regs"] = n_regs

    if gpu_ms <= 0:
        return BottleneckVerdict(kind="unknown", evidence=ev, unmeasured=unmeasured,
                                 suggests="no valid timing; nothing can be concluded")

    if empty_launch_floor_ms and gpu_ms <= empty_launch_floor_ms * th.overhead_floor_multiple:
        ev["empty_launch_floor_ms"] = empty_launch_floor_ms
        return BottleneckVerdict(
            kind="overhead_floor", evidence=ev, unmeasured=unmeasured,
            suggests="GPU time is at this GPU's empty-launch floor: the kernel body is no "
                     "longer what costs. No tiling or precision change can help; a structural "
                     "change that removes the launch entirely is the only lever.")

    if cpu_issue_ms is not None and cpu_issue_ms > 0:
        ratio = cpu_issue_ms / gpu_ms
        ev.update({"cpu_issue_ms": cpu_issue_ms, "cpu_over_gpu": round(ratio, 3)})
        if ratio >= th.launch_bound_cpu_ratio:
            return BottleneckVerdict(
                kind="launch_bound", evidence=ev, unmeasured=unmeasured,
                suggests="the CPU spends about as long ISSUING this call as the GPU spends "
                         "running it, so halving GPU time would change little a caller observes. "
                         "Fuse to remove launches, and cut per-call host work (fewer ops, no "
                         "per-call allocation). Note the harness's own timing hides most of "
                         "this cost, so the real-world gain from fusing exceeds the measured "
                         "gain.")

    # --- throughput fractions, against the ceiling that APPLIES to this kernel ---------------
    uses_tc = None
    if sass and sass.get("instructions"):
        uses_tc = bool(sass.get("tensor_core", 0) > 0)
        ev["uses_tensor_cores"] = uses_tc
    compute_ceiling = peaks.fp32_tflops if peaks else 0.0
    ceiling_name = "fp32"
    if peaks and uses_tc and peaks.tf32_tflops > 0:
        # The kernel's own instruction mix says which ceiling is the right denominator. Without
        # this a tf32 kernel is scored against fp32 and reports >100% of peak, which reads as
        # "saturated, stop optimizing" for a kernel that may have most of its headroom left.
        compute_ceiling = peaks.tf32_tflops
        ceiling_name = "tensor-core (tf32)"
    if peaks:
        ev["compute_ceiling_used"] = ceiling_name

    ai = None
    if flop_count and byte_count:
        ai = flop_count / byte_count
        ev["arithmetic_intensity"] = round(ai, 3)
    frac_bw = 0.0
    if peaks and byte_count:
        achieved_tbs = byte_count / (gpu_ms * 1e-3) / 1e12
        frac_bw = achieved_tbs / peaks.dram_tbs if peaks.dram_tbs > 0 else 0.0
        ev.update({"achieved_tbs": round(achieved_tbs, 4),
                   "pct_of_dram_peak": round(frac_bw * 100, 1)})
    frac_fl = 0.0
    if peaks and flop_count:
        achieved_fl = flop_count / (gpu_ms * 1e-3) / 1e12
        frac_fl = achieved_fl / compute_ceiling if compute_ceiling > 0 else 0.0
        ev.update({"achieved_tflops": round(achieved_fl, 3),
                   "pct_of_compute_peak": round(frac_fl * 100, 1)})
    if peaks:
        ev["ridge_flop_per_byte"] = round(peaks.ridge_flop_per_byte, 2)

    # --- cross-validation: two methods, and the analytic one wins a disagreement -------------
    # Method A: the achieved fractions above (depend on a measured ceiling).
    # Method B: the kernel's position relative to the roofline ridge (depends only on its own
    #           FLOP/byte ratio, so a throttled or contended box cannot move it).
    # KernelPro's practice, and the reasoning is sound: a busy box depresses the measured ceiling,
    # which inflates every fraction, whereas arithmetic intensity is invariant.
    disagreement = ""
    if ai is not None and peaks and peaks.ridge_flop_per_byte > 0:
        analytic_side = "compute" if ai > peaks.ridge_flop_per_byte else "memory"
        fraction_side = None
        if frac_bw >= th.dram_saturated_frac:
            fraction_side = "memory"
        elif frac_fl >= th.compute_saturated_frac:
            fraction_side = "compute"
        ev["analytic_side"] = analytic_side
        if fraction_side and fraction_side != analytic_side:
            disagreement = (
                f"the achieved fractions say {fraction_side}-bound while arithmetic intensity "
                f"{ai:.2f} vs the ridge {peaks.ridge_flop_per_byte:.1f} says {analytic_side}-side. "
                f"Deferring to the analytic bound: the fractions rest on a measured ceiling that "
                f"a contended box depresses, while the intensity ratio does not move. Treat this "
                f"verdict as low-confidence and read the evidence.")

    if frac_bw >= th.dram_saturated_frac:
        return BottleneckVerdict(
            kind="memory_bound", evidence=ev, disagreement=disagreement, unmeasured=unmeasured,
            suggests="the kernel is moving bytes at most of this GPU's measured DRAM ceiling, "
                     "so arithmetic changes cannot help. Increase reuse (larger tiles, better "
                     "blocking), or fuse to avoid writing and re-reading an intermediate.")
    if frac_fl >= th.compute_saturated_frac:
        # A fraction above 1.0 is PHYSICALLY IMPOSSIBLE and means an input is wrong, so it must
        # not be reported as "saturated, stop optimizing" -- which is what a bare threshold test
        # does with it. Two causes, and the agent needs to know which it is looking at:
        #   * the kernel uses tensor cores, so the fp32 ceiling is the wrong denominator; or
        #   * the candidate performs LESS arithmetic than the reference the FLOP count came from
        #     (an algebraic simplification, a skipped branch), so the numerator overstates its work.
        # Either way the verdict is low-confidence, and "you are at the ceiling" is false.
        impossible = ""
        if frac_fl > 1.0:
            impossible = (
                f"the kernel appears to reach {frac_fl*100:.0f}% of the {ceiling_name} ceiling, "
                f"which is impossible. Either it uses a faster arithmetic path than the ceiling "
                f"being compared against"
                + ("" if uses_tc is None else
                   (" (the instruction mix says it does NOT use tensor cores, so this is unlikely)"
                    if uses_tc is False else " (it does use tensor cores)"))
                + ", or it performs LESS arithmetic than the reference the FLOP count was measured "
                  "from -- an algebraic simplification, a skipped branch, or a shape the "
                  "candidate handles differently. Do NOT read this as 'at the ceiling': check "
                  "which of the two it is before concluding anything about headroom.")
            ev["impossible_fraction"] = round(frac_fl, 3)
        extra = ""
        if uses_tc is False and peaks and peaks.tf32_tflops > peaks.fp32_tflops:
            # Saturating fp32 while NOT using tensor cores is the most actionable verdict this
            # classifier can produce: the ceiling itself can be raised.
            extra = (f" This kernel does NOT use tensor cores, and this card's tensor-core "
                     f"ceiling is {peaks.tf32_tflops / peaks.fp32_tflops:.2f}x its fp32 one -- so "
                     f"the ceiling it is against is not the machine's ceiling. Moving the inner "
                     f"product onto tensor cores (tl.dot with a permitted precision) raises the "
                     f"limit rather than approaching it.")
        return BottleneckVerdict(
            kind="compute_bound", evidence=ev,
            disagreement=(f"{disagreement} {impossible}".strip() if impossible else disagreement),
            unmeasured=unmeasured,
            suggests=f"the kernel is near this GPU's measured {ceiling_name} ceiling. The levers "
                     f"are arithmetic: tensor cores / lower precision if the accuracy gate "
                     f"allows, and more independent accumulators for ILP." + extra)

    near_limit = []
    if n_spills:
        near_limit.append(f"spills={n_spills}")
    if n_regs and max_regs_per_thread and n_regs >= max_regs_per_thread * RESOURCE_NEAR_LIMIT_FRAC:
        near_limit.append(f"regs={n_regs}/{max_regs_per_thread}")
    if (shared_bytes and max_shared_bytes
            and shared_bytes >= max_shared_bytes * RESOURCE_NEAR_LIMIT_FRAC):
        near_limit.append(f"shared={shared_bytes}/{max_shared_bytes}")
    # Occupancy as a LEVER, which is separate from occupancy as a fact -- the fact is already in
    # `ev`, recorded at the top so every verdict carries it. On Triton this is the signal that
    # actually fires: measured on box 2, Triton's allocator caps registers and loses occupancy
    # rather than spilling (218 regs, 0 spills, 16.7% occupancy), so a spills-only test misses the
    # case entirely. The `limiter` is what makes it actionable.
    #
    # Reached only by the verdicts that fall through to here, which is what keeps a saturated
    # kernel out: `memory_bound`/`compute_bound` have already returned. The gate below repeats the
    # same requirement in its conditions, which measurement shows to be redundant -- no input
    # reaches it with either fraction above a saturation line -- so those conditions document the
    # intent rather than enforce it.
    limiter = (occupancy or {}).get("limiter")
    if occ_frac is not None and float(occ_frac) < LOW_OCCUPANCY_FRAC:
        near_limit.append(f"occupancy={float(occ_frac)*100:.0f}% (limited by {limiter})")

    if near_limit and frac_bw < th.dram_saturated_frac and frac_fl < th.compute_saturated_frac:
        ev["at_limit"] = near_limit
        lever = {
            "registers": "fewer live accumulators or a smaller BLOCK_M/BLOCK_N, so more warps "
                         "fit per SM",
            "shared_memory": "a smaller tile or fewer pipeline stages, so more blocks fit per SM",
            "warps_per_block": "more warps per block (num_warps), or more blocks",
        }.get(limiter or "", "a smaller per-thread footprint: smaller tiles, fewer live "
                             "accumulators, or different staging")
        return BottleneckVerdict(
            kind="resource_limited", evidence=ev, disagreement=disagreement,
            unmeasured=unmeasured,
            suggests=f"neither throughput ceiling is reached, but {', '.join(near_limit)} caps "
                     f"how many warps can be resident, so throughput cannot rise until the "
                     f"per-thread footprint shrinks. Try {lever}.")

    if frac_bw < th.idle_frac and frac_fl < th.idle_frac:
        return BottleneckVerdict(
            kind="latency_bound", evidence=ev, disagreement=disagreement, unmeasured=unmeasured,
            suggests="no measured resource is the limit: both throughputs are low and nothing "
                     "is at a hardware cap, which usually means too little parallelism to hide "
                     "latency, or a serial dependency. Try more programs, split-K, or deeper "
                     "pipelining. NOTE this is the residual class -- it means 'nothing we can "
                     "measure is saturated', not 'latency is proven to be the cause'.")

    return BottleneckVerdict(
        kind="mixed", evidence=ev, disagreement=disagreement, unmeasured=unmeasured,
        suggests="partially saturated on at least one ceiling without reaching the saturation "
                 "threshold on either; the cheapest next step is usually whichever fraction "
                 "above is largest.")


def _unmeasured_signals() -> list[str]:
    """Signals no verdict here rests on, because this box cannot measure them.

    Attached to EVERY verdict, including confident ones. An agent told "nothing is wrong" reasons
    differently from one told "these specific things are unknown", and only the second is true.
    """
    from kernel_optimizer.evaluation.statics import UNMEASURABLE_ON_THIS_TIER

    return list(UNMEASURABLE_ON_THIS_TIER)
