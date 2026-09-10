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
  spills          VISIBLE as STL/LDL, but as an ACCESS COUNT, which is a different quantity from
                  Triton's `n_spills` (a local-memory FOOTPRINT: bytes/4). Measured on box 1 over
                  5 kernels of rising pressure, all .32-wide: they disagree 5/5 by 4x-33x with a
                  non-constant ratio, because accesses scale with loop trip count while the
                  footprint does not. An earlier version of this note called them
                  "cross-validated, agreeing exactly" on the strength of a single kernel at
                  STL=2/LDL=1 vs n_spills=2 -- a coincidence at the smallest possible magnitude,
                  not corroboration. Both are collected; neither validates the other.
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

# Above this fraction of a measured roof, the reading is physically impossible and the honest
# conclusion is that the DENOMINATOR does not apply to this kernel -- not that the kernel is
# saturated. 1.05 rather than 1.00 because the roof is itself a measurement: the calibration
# workload does not reach the hardware's absolute maximum, so a genuinely saturated kernel can
# legitimately read a few percent above the number we measured. Nsight's own documentation notes
# that percent-of-peak metrics can exceed 100% for this reason.
#
# Applied to BOTH throughput fractions. The compute side needed it because an fp16 kernel scored
# against a tf32 ceiling read 107.8% (a missing denominator, since fixed by measuring the fp16
# ceiling). The DRAM side needs it because the byte count is LOGICAL: a kernel whose working set
# fits in L2 gets its hits counted as DRAM traffic, and a purely L2-resident streaming kernel
# measured 287% of the roof.
_IMPOSSIBLE_FRAC = 1.05

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
    # The low-precision tensor-core ceilings (P3). Without them, an fp16 or bf16 candidate was
    # scored against tf32 -- roughly half its real ceiling on a 4090 -- so L3:43's best candidate
    # read 140.7% of "peak", which reads as "saturated, stop optimizing" for a kernel with
    # perhaps 30% of its headroom unused. 13 of 25 verdicts in that run were affected, and it got
    # worse as candidates improved, because every candidate good enough to lead was low-precision.
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0

    def compute_ceiling_for(self, precision: str | None) -> tuple[float, str]:
        """The arithmetic ceiling that applies to a kernel computing in `precision`.

        Falls back along a chain rather than to zero: an unmeasured fp16 ceiling should give the
        tf32 figure (wrong but flagged by `impossible_fraction`) rather than silently disabling
        the compute test. The returned name says which ceiling was used, so the evidence records
        what the percentage is a percentage OF.
        """
        if precision == "fp16" and self.fp16_tflops > 0:
            return self.fp16_tflops, "tensor-core (fp16)"
        if precision == "bf16" and self.bf16_tflops > 0:
            return self.bf16_tflops, "tensor-core (bf16)"
        if self.tf32_tflops > 0:
            return self.tf32_tflops, "tensor-core (tf32)"
        return self.fp32_tflops, "fp32 (no tensor cores)"

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
    overhead_gpu_ms: float | None = None,
    precision: str | None = None,
    # --- per-candidate cost (G1/G2/G4/G6) -------------------------------------------------
    # `flop_count` and `byte_count` above are TASK-level: they come from TaskCost, measured once
    # on the reference, identical for every candidate. That is right for the question they answer
    # -- "can this task ever be compute-bound on this card", which no single-candidate
    # measurement can answer -- but it makes `pct_of_dram_peak` equal to
    # task_constant / gpu_ms, i.e. 1/latency on a different scale. Verified across 5 runs:
    # gpu_ms varies up to 4.6x between candidates while the derived numerator varies <=0.36%.
    #
    # The parameters below are the per-candidate counterpart. They are reported as MEASUREMENTS
    # in their own right and are never divided by gpu_ms, precisely so they cannot degenerate the
    # same way. Every one was checked to vary across candidates before being plumbed here.
    peak_alloc_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
    candidate_aten_bytes: int | None = None,
    candidate_aten_ops: int | None = None,
    threads_launched: int | None = None,
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

    # G5: capacity as a QUANTITY, not only as a boolean. Registers and shared memory were
    # reported solely through the `at_limit` strings below, which say "shared=49664/101376" when a
    # kernel is near the limit and say NOTHING when it is not. So an agent asking the most basic
    # trading question -- "how much shared memory do I have left to spend on a bigger tile?" --
    # had no answer, and "spend capacity to cut traffic" is among the most common structural moves
    # available (L3:43's second-round rewrite computed 73728 -> 40960 bytes by hand precisely to
    # unlock a larger tile).
    #
    # Reported for every verdict, not just resource_limited: the headroom matters most when the
    # kernel is memory_bound, since that is when spending capacity to buy reuse is the move.
    # Absolute bytes AND the remaining fraction, because the agent needs both -- a fraction alone
    # cannot be turned into a tile size.
    if shared_bytes is not None and max_shared_bytes:
        ev["shared_headroom_bytes"] = max(0, max_shared_bytes - shared_bytes)
        ev["shared_used_frac"] = round(shared_bytes / max_shared_bytes, 3)
    if n_regs is not None and max_regs_per_thread:
        ev["reg_headroom_per_thread"] = max(0, max_regs_per_thread - n_regs)
        ev["reg_used_frac"] = round(n_regs / max_regs_per_thread, 3)

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
        # P2 cause (c): the ratio's denominator must be the overhead probe's OWN gpu_ms, not
        # the harness's headline latency. The probe measures both numbers in one pass with no
        # L2 flush between calls, and its docstring says plainly that its gpu_ms is warm-cache
        # and usable ONLY as this ratio's denominator. `ProfileRecord.cpu_over_gpu` honours
        # that; this function used to divide by the harness's `gpu_ms` instead.
        #
        # The bias is NOT in a fixed direction -- an earlier note here claimed it always
        # under-detected, and measurement disproved that. The two figures differ because the
        # harness flushes L2 between calls (adding cost) while the probe lets many small
        # launches queue (hiding cost), and which effect dominates depends on the kernel:
        #   L3:43 theta_best   harness 3.0126 ms vs probe 3.0050 ms  -- 0.25% apart
        #   40 tiny elementwise harness 0.3497 ms vs probe 0.5849 ms -- harness 40% SMALLER
        # So the mismatch is a wrong number of unpredictable sign, not a known-direction skew.
        # Falls back to gpu_ms only when the probe's figure is absent, and labels which
        # denominator produced the ratio so a reader can tell a sound one from a fallback.
        use_probe = bool(overhead_gpu_ms and overhead_gpu_ms > 0)
        denom = overhead_gpu_ms if use_probe else gpu_ms
        ratio = cpu_issue_ms / denom
        ev.update({"cpu_issue_ms": cpu_issue_ms, "cpu_over_gpu": round(ratio, 3),
                   "cpu_over_gpu_denominator": ("overhead_probe_gpu_ms" if use_probe
                                                else "harness_gpu_ms_FALLBACK")})
        if use_probe:
            ev["overhead_gpu_ms"] = overhead_gpu_ms
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
    if peaks and uses_tc:
        # The kernel's own instruction mix says it uses tensor cores; `precision` says WHICH
        # tensor-core ceiling applies. Without the precision this always used tf32, and an fp16
        # kernel -- roughly 2x tf32 on a 4090 -- read as >100% of peak, i.e. "saturated, stop
        # optimizing" for a kernel with headroom left. Falls back through tf32 to fp32 when the
        # matching ceiling was not measured, so an older calibration still classifies.
        compute_ceiling, ceiling_name = peaks.compute_ceiling_for(precision)
    if peaks:
        ev["compute_ceiling_used"] = ceiling_name
        if precision:
            ev["candidate_precision"] = precision

    ai = None
    if flop_count and byte_count:
        ai = flop_count / byte_count
        ev["arithmetic_intensity"] = round(ai, 3)
    frac_bw = 0.0
    if peaks and byte_count:
        achieved_tbs = byte_count / (gpu_ms * 1e-3) / 1e12
        frac_bw = achieved_tbs / peaks.dram_tbs if peaks.dram_tbs > 0 else 0.0
        ev.update({"achieved_tbs": round(achieved_tbs, 4),
                   "pct_of_dram_peak": round(frac_bw * 100, 1),
                   # Say out loud what this number is, in the evidence the agent reads. It is
                   # computed from a TASK-level byte count, so within one task it is a rescaling
                   # of 1/gpu_ms and orders candidates exactly as latency does. It answers "how
                   # far is this kernel from the card's physical roof" -- a real and useful
                   # question, which is why it stays -- and it does NOT answer "how much traffic
                   # did this candidate do", nor may it count as a dimension independent of
                   # latency in any multi-binding tally.
                   "dram_pressure_basis": "task-level compulsory bytes / gpu_ms; within a task "
                                          "this is 1/latency rescaled, not a per-candidate "
                                          "traffic measurement"})
    frac_fl = 0.0
    if peaks and flop_count:
        achieved_fl = flop_count / (gpu_ms * 1e-3) / 1e12
        frac_fl = achieved_fl / compute_ceiling if compute_ceiling > 0 else 0.0
        ev.update({"achieved_tflops": round(achieved_fl, 3),
                   "pct_of_compute_peak": round(frac_fl * 100, 1),
                   "compute_pressure_basis": "task-level FLOP count / gpu_ms; same caveat as "
                                             "dram_pressure_basis"})

    # --- per-candidate cost: measured, never divided by latency (G1/G2/G4/G6) ----------------
    # Reported as absolute quantities. Comparing them across candidates is the point; dividing
    # them by gpu_ms is exactly the mistake the two fractions above embody.
    if peak_alloc_bytes is not None:
        ev["peak_alloc_mib"] = round(peak_alloc_bytes / 2**20, 1)
    if peak_reserved_bytes is not None:
        ev["peak_reserved_mib"] = round(peak_reserved_bytes / 2**20, 1)
    if candidate_aten_bytes is not None:
        ev["candidate_aten_mib"] = round(candidate_aten_bytes / 2**20, 1)
        # The bound is stated with the number, not in a doc somewhere else. A reader who takes
        # this for the candidate's DRAM traffic will conclude a well-fused kernel moves almost
        # nothing, which is the opposite of true.
        ev["candidate_aten_basis"] = ("LOWER bound on this candidate's traffic: aten-level "
                                      "materialization only, blind to anything fused inside a "
                                      "kernel. Upper bound is the task's reference_bytes.")
    if candidate_aten_ops is not None:
        ev["candidate_aten_ops"] = candidate_aten_ops
    if threads_launched is not None:
        ev["threads_launched"] = threads_launched
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
        # G7: a DRAM fraction above 1.0 is PHYSICALLY IMPOSSIBLE and means the denominator does
        # not apply to this kernel, so it must not be reported as "saturated, stop optimizing".
        # The compute branch below has always handled its own impossible case; this one did not,
        # and a positive control caught it: five kernels whose bottleneck is known by construction
        # were classified, and the two whose working set FITS IN L2 were mislabelled -- one
        # purely L2-resident streaming kernel read 287% of the DRAM roof.
        #
        # The cause is that `byte_count` counts LOGICAL bytes. An L2 hit is counted as traffic and
        # never crosses the memory bus, so the fraction inflates without bound as the working set
        # shrinks below the 72 MiB L2. Reporting `unknown` is strictly stronger than clamping to
        # 1.0: clamping would still say "at the roof", which is the wrong action, while the true
        # statement is that this quantity cannot be evaluated for this kernel.
        #
        # DORMANT ON TODAY'S TASKS, and that is why this is a fix and not an emergency: all three
        # real tasks are far above L2 (L3:21 307 MiB, L3:43 397 MiB, L3:48 1.351 GB), so no
        # current classification is affected. It wakes as soon as the task range widens to the
        # smaller level1/level2 problems.
        if frac_bw > _IMPOSSIBLE_FRAC:
            ev["impossible_dram_fraction"] = round(frac_bw * 100, 1)
            return BottleneckVerdict(
                kind="unknown", evidence=ev, disagreement=disagreement, unmeasured=unmeasured,
                suggests=(
                    f"the measured DRAM fraction is {frac_bw * 100:.0f}% of this card's roof, "
                    f"which is physically impossible, so the traffic figure does not describe "
                    f"this kernel and no bandwidth verdict can be given. The usual cause is a "
                    f"working set that fits in L2: the byte count is logical, and an L2 hit is "
                    f"counted as traffic without crossing the memory bus. Compare the task's "
                    f"working set against this card's L2 before treating bandwidth as the limit."))
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
            # P3 note: the leading cause of this used to be a MISSING ceiling -- an fp16 kernel
            # scored against tf32 -- which produced 13 of 25 impossible fractions on L3:43. The
            # fp16/bf16 ceilings are measured now, so a fraction above 1.0 with a matching
            # ceiling name is much more likely to be the second cause. Say which ceiling was
            # used, so the reader can tell a wrong denominator from a wrong numerator.
            unmatched = ceiling_name != {"fp16": "tensor-core (fp16)",
                                         "bf16": "tensor-core (bf16)"}.get(precision or "",
                                                                           ceiling_name)
            impossible = (
                f"the kernel appears to reach {frac_fl*100:.0f}% of the {ceiling_name} ceiling, "
                f"which is impossible. Either it uses a faster arithmetic path than the ceiling "
                f"being compared against"
                + ("" if uses_tc is None else
                   (" (the instruction mix says it does NOT use tensor cores, so this is unlikely)"
                    if uses_tc is False else " (it does use tensor cores)"))
                + (f" -- and note this box has no measured {precision} ceiling, so the "
                   f"{ceiling_name} figure was substituted, which is very likely the whole "
                   f"explanation" if unmatched and precision in ("fp16", "bf16") else "")
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
