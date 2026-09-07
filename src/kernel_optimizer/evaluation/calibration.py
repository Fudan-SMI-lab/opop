"""Self-calibration: derive this box's ceilings and classification thresholds by measurement.

WHY THIS EXISTS. The bottleneck classifier needs numbers to compare against: how many bytes per
second is "saturating DRAM", how many FLOP/s is "near the compute ceiling", how long is "just the
launch floor". Hardcoding those is wrong twice over.

  1. They are not portable. This harness has already run on an RTX 5080 Laptop (sm_120, 16GB) and
     an RTX 4090 (sm_89, 24GB), and the plan is to run on more. A constant tuned on one is a
     silent misclassifier on the next -- and the failure is invisible, because a wrong verdict
     still reads like a verdict.
  2. They are not even portable across CONTAINERS on one card. Rented boxes cap clocks and share
     the PCIe root; the same 4090 can present a materially different achievable bandwidth on two
     different days. A datasheet number is not what a kernel can get.

So the ceilings are MEASURED on the box, and every threshold in the classifier is a dimensionless
fraction of a measured ceiling. Moving to a new GPU changes the ceilings and changes nothing in
the code.

WHAT IS MEASURED, and why each one:
  dram_tbs     -- a 512 MB streaming add. The largest bandwidth a trivially-parallel kernel gets.
  fp32_tflops  -- an 8192^3 fp32 matmul with tf32 OFF. The fp32 (non-tensor-core) ceiling.
  tf32_tflops  -- the same matmul with tf32 ON. Kept separate because a kernel using tensor cores
                  compared against the fp32 ceiling reads as >100% of peak, and a kernel NOT
                  using them compared against the tf32 ceiling reads as hopeless. The classifier
                  needs to know which ceiling applies.
  empty_launch_floor_ms -- the cost of a launch that does nothing. This is the input
                  `overhead_floor` has been missing: without it, a kernel whose body is free is
                  classified by its throughput fractions, which are near zero, so it lands in
                  `latency_bound` and the agent is told to add parallelism to a kernel that is
                  already at the floor.

THRESHOLD DERIVATION IS ALSO MEASURED, not guessed. The four yardstick workloads have
analytically-known bottlenecks (a 4096^3 matmul IS compute-bound; a 512 MB copy IS
memory-bound). So the saturation thresholds are placed BELOW what the known-saturated workload
achieves and ABOVE what the known-unsaturated ones achieve -- i.e. derived from the separation
that the box itself exhibits. Two guessed constants were disproved this way on the 4090:

    COMPUTE_SATURATED_FRAC = 0.50  -- the matmul reaches 97.5% of the fp32 ceiling. A 0.50 line
                                     would call a half-speed kernel "compute bound" and stop the
                                     agent from optimizing it.
    LAUNCH_BOUND_CPU_RATIO = 1.0   -- the indisputably launch-bound workload (40 tiny ops)
                                     measures 0.973, i.e. BELOW the line. The test as written
                                     judges backwards and would never fire.

`derive_thresholds` is a pure function of the measurements so it is testable without a GPU, and
`Calibration` is content-addressed by device identity so a cached calibration is never silently
reused on a different card.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# How far below the known-saturated workload's achievement to put the saturation line. The
# yardstick tells us what saturation looks like ON THIS BOX; this margin is the only judgement
# call, and it is dimensionless so it travels. 0.85 means "within 15% of what a kernel that is
# definitely saturated actually got".
SATURATION_MARGIN = 0.85
# How far above the known-unsaturated workloads to put the "nothing is saturated" line. Wide,
# because the residual class must not swallow kernels that are partially saturated.
IDLE_HEADROOM = 2.0
# A measured ceiling this far below the derived spec figure means the card was throttled or
# contended while calibrating. Not fatal -- the run continues -- but the calibration is flagged
# so a verdict resting on it can be discounted.
SUSPECT_BELOW_SPEC_FRAC = 0.60


class Yardstick(BaseModel):
    """One workload whose bottleneck is known analytically, as measured on this box."""

    model_config = ConfigDict(frozen=True)

    name: str
    # What this workload IS, decided analytically before measuring: the ground truth the
    # thresholds must reproduce. Not a prediction of the classifier -- an input to placing its
    # lines. "unsaturated" means neither ceiling is near reached, which is the honest label for a
    # realistic fused op; it deliberately does NOT reuse the classifier's `mixed` name, which
    # means "partially saturated" and is a different claim.
    truth: str          # "compute" | "memory" | "launch" | "unsaturated"
    gpu_ms: float
    cpu_issue_ms: float
    flop_count: int
    byte_count: int

    @property
    def cpu_over_gpu(self) -> float:
        return self.cpu_issue_ms / self.gpu_ms if self.gpu_ms > 0 else 0.0

    def pct_of_dram(self, dram_tbs: float) -> float:
        if dram_tbs <= 0 or self.gpu_ms <= 0:
            return 0.0
        return (self.byte_count / (self.gpu_ms * 1e-3) / 1e12) / dram_tbs

    def pct_of_fp32(self, fp32_tflops: float) -> float:
        if fp32_tflops <= 0 or self.gpu_ms <= 0:
            return 0.0
        return (self.flop_count / (self.gpu_ms * 1e-3) / 1e12) / fp32_tflops


class Thresholds(BaseModel):
    """Dimensionless classification lines, derived from this box's own separation."""

    model_config = ConfigDict(frozen=True)

    dram_saturated_frac: float
    compute_saturated_frac: float
    idle_frac: float
    launch_bound_cpu_ratio: float
    resource_near_limit_frac: float = 0.80
    overhead_floor_multiple: float = 1.15

    # Which yardstick separations produced each line, so a threshold can be audited rather than
    # trusted. A number whose provenance is unrecorded is indistinguishable from a guess.
    derivation: dict = Field(default_factory=dict)


class Calibration(BaseModel):
    """Everything the classifier needs, measured on one box at one time."""

    model_config = ConfigDict(frozen=True)

    device_name: str
    capability: list[int]
    sm_count: int
    # Identity the cache is keyed on. A calibration is only reusable on the same card with the
    # same driver and torch: a driver upgrade can move achievable bandwidth.
    torch_version: str = ""
    driver_version: str = ""

    dram_tbs: float
    fp32_tflops: float
    tf32_tflops: float = 0.0
    empty_launch_floor_ms: float = 0.0

    spec_dram_tbs: float = 0.0
    l2_bytes: int = 0
    yardsticks: list[Yardstick] = Field(default_factory=list)
    thresholds: Thresholds | None = None

    # Populated when a measured ceiling falls far short of the derived spec figure: the numbers
    # are still used (they are what this box gives) but every verdict resting on them is
    # discountable, and the operator is told to recalibrate on an idle box.
    suspect: list[str] = Field(default_factory=list)
    measured_at: str = ""

    @property
    def ridge_flop_per_byte(self) -> float:
        """Roofline ridge: below it a kernel is memory-side, above it compute-side."""
        if self.dram_tbs <= 0:
            return 0.0
        return self.fp32_tflops / self.dram_tbs

    @property
    def tf32_ridge_flop_per_byte(self) -> float:
        """The ridge that applies to a kernel using tensor cores. Materially different: on the
        4090 the tf32 ceiling is several times the fp32 one, so a kernel judged compute-bound
        against fp32 may have most of its headroom left on the tensor-core path."""
        if self.dram_tbs <= 0 or self.tf32_tflops <= 0:
            return 0.0
        return self.tf32_tflops / self.dram_tbs

    def identity(self) -> str:
        return "|".join([self.device_name, ".".join(str(c) for c in self.capability),
                         str(self.sm_count), self.torch_version, self.driver_version])


def derive_thresholds(
    yardsticks: list[Yardstick],
    dram_tbs: float,
    fp32_tflops: float,
) -> Thresholds:
    """Place the classification lines from the separation this box actually exhibits.

    Pure function of the measurements: no GPU, no I/O, fully testable. Each line is placed
    relative to what the known-saturated yardstick achieved, with a dimensionless margin, and
    each falls back to a documented default when its yardstick is missing rather than silently
    producing a line derived from nothing.

    The FALLBACKS matter as much as the derivations. If the compute yardstick did not run, the
    honest default is a HIGH bar (0.80), because the measured evidence says a genuinely
    compute-bound kernel reaches ~0.97 -- guessing low is what makes the classifier declare
    victory on a kernel with half its headroom left.
    """
    by_truth = {y.truth: y for y in yardsticks}
    derivation: dict = {}

    mem = by_truth.get("memory")
    if mem is not None:
        achieved = mem.pct_of_dram(dram_tbs)
        dram_frac = max(0.30, min(0.95, achieved * SATURATION_MARGIN))
        derivation["dram_saturated_frac"] = (
            f"{mem.name} is memory-bound by construction and reached {achieved*100:.1f}% of the "
            f"measured DRAM ceiling; the line sits at {SATURATION_MARGIN:.2f} of that")
    else:
        dram_frac = 0.60
        derivation["dram_saturated_frac"] = "no memory yardstick; documented default 0.60"

    comp = by_truth.get("compute")
    if comp is not None:
        achieved = comp.pct_of_fp32(fp32_tflops)
        compute_frac = max(0.30, min(0.95, achieved * SATURATION_MARGIN))
        derivation["compute_saturated_frac"] = (
            f"{comp.name} is compute-bound by construction and reached {achieved*100:.1f}% of "
            f"the measured fp32 ceiling; the line sits at {SATURATION_MARGIN:.2f} of that. A "
            f"guessed 0.50 would have called a half-speed kernel saturated")
    else:
        compute_frac = 0.80
        derivation["compute_saturated_frac"] = (
            "no compute yardstick; default 0.80 -- deliberately HIGH, since measured evidence "
            "puts a genuinely compute-bound kernel near 0.97")

    # The launch line comes from the launch-bound yardstick, not from a round number. Measured
    # 0.973 on the 4090 for a workload that is indisputably launch-bound, so a line at 1.0 can
    # never fire. Placed just below what that workload exhibits.
    lb = by_truth.get("launch")
    others = [y for y in yardsticks if y.truth != "launch"]
    if lb is not None:
        ratio = lb.cpu_over_gpu
        # Must also stay above the non-launch workloads, or a compute kernel with a chatty host
        # side gets called launch-bound. The mixed yardstick is the binding one in practice
        # (0.711 on the 4090), so the line is the midpoint when they are close.
        highest_other = max((y.cpu_over_gpu for y in others), default=0.0)
        line = ratio * 0.90
        if line <= highest_other:
            line = (ratio + highest_other) / 2.0 if ratio > highest_other else ratio * 0.99
        launch_ratio = max(0.30, min(2.0, line))
        derivation["launch_bound_cpu_ratio"] = (
            f"{lb.name} is launch-bound by construction and measures cpu/gpu {ratio:.3f}; the "
            f"highest non-launch yardstick is {highest_other:.3f}, so the line is {launch_ratio:.3f}. "
            f"A guessed 1.0 judges BACKWARDS -- the launch-bound workload is below it")
    else:
        launch_ratio = 0.85
        derivation["launch_bound_cpu_ratio"] = (
            "no launch yardstick; default 0.85 -- NOT 1.0, which measurement showed to be above "
            "what a launch-bound workload exhibits")

    # The idle line must sit above every yardstick that is NOT saturated on that axis, or the
    # residual class swallows partially-saturated kernels. Derived from the largest such
    # achievement rather than from a round fraction.
    unsat_dram = [y.pct_of_dram(dram_tbs) for y in yardsticks if y.truth != "memory"]
    unsat_comp = [y.pct_of_fp32(fp32_tflops) for y in yardsticks if y.truth != "compute"]
    worst_unsat = max(unsat_dram + unsat_comp) if (unsat_dram or unsat_comp) else 0.0
    idle_frac = max(0.05, min(0.40, worst_unsat * IDLE_HEADROOM))
    derivation["idle_frac"] = (
        f"the highest fraction reached by a yardstick that is NOT saturated on that axis is "
        f"{worst_unsat*100:.1f}%; the idle line is {IDLE_HEADROOM:.1f}x that, so a partially "
        f"saturated kernel is not swallowed by the residual class")

    return Thresholds(
        dram_saturated_frac=round(dram_frac, 4),
        compute_saturated_frac=round(compute_frac, 4),
        idle_frac=round(idle_frac, 4),
        launch_bound_cpu_ratio=round(launch_ratio, 4),
        derivation=derivation,
    )


def flag_suspect(dram_tbs: float, spec_dram_tbs: float) -> list[str]:
    """Report ceilings that fall far short of spec: the box was busy or throttled.

    Deliberately does NOT reject the calibration. A throttled box is still the box the run
    happens on, and its achievable bandwidth is the honest denominator. What must not happen is
    a verdict resting on a bad ceiling being reported with the same confidence as one resting on
    a good one.
    """
    out: list[str] = []
    if spec_dram_tbs > 0 and dram_tbs > 0:
        frac = dram_tbs / spec_dram_tbs
        if frac < SUSPECT_BELOW_SPEC_FRAC:
            out.append(
                f"measured DRAM ceiling {dram_tbs:.3f} TB/s is only {frac*100:.0f}% of the "
                f"derived spec {spec_dram_tbs:.3f} TB/s: the GPU was likely throttled or shared "
                f"during calibration. Verdicts using % of DRAM peak are inflated. Recalibrate "
                f"on an idle box with --recalibrate.")
    return out


def cache_path(run_root: Path) -> Path:
    """Where a calibration is cached. Beside the runs rather than inside one: it describes the
    BOX, not the run, and re-measuring it per run would cost minutes of GPU time each."""
    return run_root / "calibration.json"


def load_cached(path: Path, identity: str | None = None) -> Calibration | None:
    """Load a cached calibration, refusing one measured on different hardware.

    The identity check is the point. A cache keyed only on the file path would be silently
    reused after a box change -- and since every classification is a fraction of these
    ceilings, that produces confident verdicts computed against another card's limits.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cal = Calibration.model_validate(data)
    except Exception:  # noqa: BLE001 — a corrupt cache must re-measure, not crash a run
        return None
    if identity is not None and cal.identity() != identity:
        return None
    return cal


def save(path: Path, cal: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal.model_dump(), f, indent=2, ensure_ascii=False)
