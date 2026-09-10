"""S2: one independent verdict per resource dimension, instead of one label per candidate.

WHAT IS WRONG WITH THE LABEL. `bottleneck.classify()` walks an if/else chain and returns a single
`kind`. Measured in production: on run-l3-21-20260908-232211, **19 of 20** candidates got the same
label (`resource_limited`); on run-l3-48-20260909-115701, 12 of 17 got `memory_bound`. The defect is
not that the chain picks the wrong winner -- it is that a candidate against two walls at once has no
way to say so, because the return type holds one string.

The quantities are already there. `classify()`'s `evidence` dict carries occupancy, its limiter,
n_regs, n_spills, shared headroom and used fraction, the achieved fractions, arithmetic intensity,
tensor-core usage, per-candidate allocation and aten traffic, and the backend-reachable fraction.
**The vector exists and the if/else chain collapses it into a label.** This module stops collapsing
it.

THE RECORD, one per dimension (plan §1.5.3):

    {dimension_id, measured, ceiling, ceiling_provenance, verdict, confidence, applicable}

Four rules, each of which exists because of a specific measured failure:

1. `applicable` is a first-class field, never a default. A card without bf16 does not have that
   dimension; it does not have it "at zero". `applicable=false` and `measured=0` must stay
   distinguishable -- the same discipline `task_cost.py` already applies to `flop_count=0` (legal
   for maxpool, a failure for matmul).

2. Verdicts stay plural. No vote, no weighting, no max. When three dimensions are binding the
   output is three binding records. This is the actual fix.

3. Occupancy's polarity is inverted and every dimension declares its own. Treating "high = binding"
   uniformly reported ZERO multi-dimension binding while a counter-example sat in the input, because
   occupancy binds when it is LOW. The polarity table is `conversion._DIMENSIONS`, reused here
   rather than restated -- one vocabulary for state, expectations, reconciliation and reporting.

4. A ceiling's reachability is derived, never assumed. Roofline's own paper overturns its ceiling
   order on SpMV, and its definition of the upper ceilings includes work "inherently lacking in a
   kernel". That is exactly the L3:48 accident: the tensor-core roof was unreachable on that task,
   the framework kept aiming at it, 8 of 8 tensor-core candidates were rejected and 7 of 7 scalar
   ones accepted, and the winning 2.09 ms / 8.90x used no tensor cores at all. So a ceiling carries
   its provenance and a reachable ceiling is distinguished from a nominal one.

SEVERITY IS THREE-VALUED, not two. `binding / near-binding / slack`. A binary split mis-routes the
middle: L3:48 at 90.4% of its measured DRAM roof and L3:43 with 69x fusion headroom only separate
under three bands.

WHAT THIS MODULE DOES NOT DO. It does not touch `classify()`. The label path stays exactly as it is,
because `v3.diagnosis.mode` has to be switchable for the control run -- and because the vector is
written to events.jsonl either way. Recording costs nothing and is not what carries risk; what
carries risk is what reaches the PROMPT, which is the one thing that has an external
counter-example (few-shot exemplars measurably LOWERED KernelBench L1 fast_1 from 10% to 6%).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.evaluation.conversion import _DIMENSIONS

Verdict = Literal["binding", "near-binding", "slack", "not-applicable", "unknown"]

# Bands. `binding` at or above the saturation line, `near-binding` in the band below it, `slack`
# under that. Two thresholds rather than one because of the three-way split above.
#
# These are FRACTIONS OF THE DIMENSION'S OWN CEILING, so they are meaningful per dimension and never
# compared across dimensions -- comparing normalised movements across dimensions is banned
# (measured: shared memory's median swing is 13.5x registers', purely from the normalisation, which
# turned an "argmax |delta|" rule into "always pick shared", 61.5% of the time).
BINDING_FRAC = 0.90
NEAR_BINDING_FRAC = 0.70

# Occupancy is the inverted one: it binds when LOW. A separate pair, because "90% of the occupancy
# ceiling" would mean the opposite of binding.
OCC_BINDING_BELOW = 0.30
OCC_NEAR_BINDING_BELOW = 0.50


class DimensionRecord(BaseModel):
    """One dimension's state on one candidate. Plural by construction: never merged, never ranked."""

    model_config = ConfigDict(frozen=True)

    dimension_id: str
    # None when the dimension applies but was not measured on this box/candidate. Distinct from
    # `applicable=False`, which means the dimension does not exist here at all.
    measured: float | None = None
    ceiling: float | None = None
    # Where the ceiling came from, in words. Never a bare number: the L3:48 accident was a nominal
    # tensor-core roof treated as a target on a task that could not reach it.
    ceiling_provenance: str = ""
    verdict: Verdict = "unknown"
    # 0..1. Lower when a reading is a bound rather than a value (aten traffic under-reports as
    # fusion improves) or when the ceiling is nominal rather than measured on this box.
    confidence: float = 1.0
    applicable: bool = True
    # Why the dimension does not apply, when it does not. Required in that case -- an
    # `applicable=False` with no reason is indistinguishable from a collection failure.
    not_applicable_reason: str = ""
    # True when higher is better (occupancy). Carried on the record so a consumer cannot get the
    # polarity wrong by looking it up in the wrong place.
    higher_is_better: bool = False
    unit: str = ""

    def is_binding(self) -> bool:
        return self.applicable and self.verdict == "binding"


class DimensionState(BaseModel):
    """Every dimension's record for one candidate. The state view of plan step 1."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str = ""
    # Keyed by (candidate, structural_signature) upstream: resource use is a property of the
    # STRUCTURE, so a rewrite must be re-measured rather than inheriting its parent's vector.
    structural_signature: str = ""
    records: tuple[DimensionRecord, ...] = ()

    def binding(self) -> tuple[DimensionRecord, ...]:
        return tuple(r for r in self.records if r.is_binding())

    def applicable(self) -> tuple[DimensionRecord, ...]:
        return tuple(r for r in self.records if r.applicable)


def _band(frac: float | None, higher_is_better: bool) -> Verdict:
    """Which of the three bands a fraction-of-ceiling falls in. `None` -> unknown."""
    if frac is None:
        return "unknown"
    if higher_is_better:
        # Inverted: binding when the value is LOW. Uses its own thresholds because "90% of the
        # occupancy ceiling" would read as binding when it is the opposite.
        if frac < OCC_BINDING_BELOW:
            return "binding"
        if frac < OCC_NEAR_BINDING_BELOW:
            return "near-binding"
        return "slack"
    if frac >= BINDING_FRAC:
        return "binding"
    if frac >= NEAR_BINDING_FRAC:
        return "near-binding"
    return "slack"


def _record(dimension_id: str, measured: float | None, ceiling: float | None,
            provenance: str, *, applicable: bool = True, not_applicable_reason: str = "",
            confidence: float = 1.0) -> DimensionRecord:
    meta = _DIMENSIONS.get(dimension_id, {})
    higher = meta.get("lower_is_better") is False
    unit = str(meta.get("unit", ""))

    if not applicable:
        # A dimension that does not apply carries no verdict and MUST carry a reason. Returning
        # `slack` here would be the worst of the options -- it reads as "there is headroom".
        return DimensionRecord(
            dimension_id=dimension_id, measured=measured, ceiling=ceiling,
            ceiling_provenance=provenance, verdict="not-applicable", confidence=confidence,
            applicable=False,
            not_applicable_reason=not_applicable_reason or "declared inapplicable without a reason",
            higher_is_better=higher, unit=unit)

    frac: float | None = None
    if measured is not None and ceiling:
        frac = measured / ceiling
    elif measured is not None and higher:
        # Occupancy is already a fraction; it has no separate ceiling to divide by.
        frac = measured

    return DimensionRecord(
        dimension_id=dimension_id, measured=measured, ceiling=ceiling,
        ceiling_provenance=provenance, verdict=_band(frac, higher),
        confidence=confidence, applicable=True, higher_is_better=higher, unit=unit)


def state_from_evidence(evidence: dict[str, Any], *, candidate_id: str = "",
                        structural_signature: str = "",
                        device_limits: dict[str, Any] | None = None) -> DimensionState:
    """Build the per-dimension state from the evidence `classify()` already computes.

    Deliberately a READER of the existing evidence rather than a second measurement path. Two
    reasons, both from experience: a parallel path would drift from the one the label uses and make
    the control run uninterpretable, and every quantity here has already been checked to vary across
    candidates -- a re-derivation could silently reintroduce a task-level constant.

    `evidence` keys are read through explicit names with no fallback guessing. A key that moved is
    reported as an unmeasured dimension, never silently as zero: guessing field paths and reading
    the miss as a zero result is a failure this project has hit repeatedly.
    """
    dev = device_limits or {}
    out: list[DimensionRecord] = []

    # --- occupancy: the inverted dimension, and the one that actually fires on Triton ----------
    occ = evidence.get("occupancy")
    if isinstance(occ, (int, float)):
        limiter = evidence.get("occupancy_limiter") or ""
        out.append(_record(
            "occupancy", float(occ), 1.0,
            "measured after one launch" + (f"; limiter={limiter}" if limiter else "")))
    else:
        out.append(_record("occupancy", None, 1.0, "not measured on this candidate"))

    # --- registers ------------------------------------------------------------------------------
    regs = evidence.get("n_regs")
    max_regs = dev.get("max_regs_per_thread")
    if isinstance(regs, (int, float)) and max_regs:
        out.append(_record("n_regs", float(regs), float(max_regs),
                           "hardware limit per thread, from this box's device query"))
    else:
        out.append(_record("n_regs", None, float(max_regs) if max_regs else None,
                           "register count not read from the compiler"))

    # --- spills: a count, and its ceiling is zero, so the band logic does not apply -------------
    spills = evidence.get("n_spills")
    if isinstance(spills, (int, float)):
        # Any spill at all is binding; there is no "70% of no spills".
        out.append(DimensionRecord(
            dimension_id="n_spills", measured=float(spills), ceiling=0.0,
            ceiling_provenance="a spill is binding by nature: the ceiling is zero spills",
            verdict="binding" if float(spills) > 0 else "slack",
            higher_is_better=False, unit=str(_DIMENSIONS["n_spills"]["unit"])))
    else:
        out.append(_record("n_spills", None, 0.0, "spill count not read from the compiler"))

    # --- shared memory --------------------------------------------------------------------------
    shared_frac = evidence.get("shared_used_frac")
    max_shared = dev.get("max_shared_bytes_optin") or dev.get("max_shared_bytes")
    if isinstance(shared_frac, (int, float)) and max_shared:
        out.append(_record("shared_bytes", float(shared_frac) * float(max_shared),
                           float(max_shared),
                           "per-block opt-in limit from this box's device query"))
    else:
        out.append(_record("shared_bytes", None,
                           float(max_shared) if max_shared else None,
                           "shared-memory usage not read from the compiler"))

    # --- per-candidate memory footprint ---------------------------------------------------------
    # `peak_alloc_mib` is a real per-candidate measurement (jumps 9.6-41.3% between candidates),
    # but its ceiling is the card's memory, which these tasks are nowhere near -- so it is reported
    # with a ceiling and will land in `slack`, which is the honest reading rather than a hidden one.
    peak_mib = evidence.get("peak_alloc_mib")
    vram_gb = dev.get("vram_gb")
    if isinstance(peak_mib, (int, float)) and vram_gb:
        out.append(_record("peak_alloc_bytes", float(peak_mib) * 2**20,
                           float(vram_gb) * 2**30,
                           "this box's total VRAM from the device query"))
    else:
        out.append(_record("peak_alloc_bytes", None, None, "peak allocation not measured"))

    # --- aten-level traffic: a LOWER BOUND, so its confidence is reduced on purpose -------------
    aten_mib = evidence.get("candidate_aten_mib")
    if isinstance(aten_mib, (int, float)):
        out.append(_record(
            "candidate_aten_bytes", float(aten_mib) * 2**20, None,
            "aten-level lower bound: a fused kernel under-reports, so this cannot bound a ceiling",
            confidence=0.5))
    else:
        out.append(_record("candidate_aten_bytes", None, None,
                           "aten traffic not measured", confidence=0.5))

    aten_ops = evidence.get("candidate_aten_ops")
    if isinstance(aten_ops, (int, float)):
        out.append(_record("candidate_aten_ops", float(aten_ops), None,
                           "aten op count; a lower bound for the same reason as the byte count",
                           confidence=0.5))
    else:
        out.append(_record("candidate_aten_ops", None, None, "aten op count not measured",
                           confidence=0.5))

    # --- threads launched: no polarity, so no band can ever apply -------------------------------
    # `applicable` and the verdict `not-applicable` answer DIFFERENT questions, and this dimension is
    # where they come apart. The dimension exists here and was measured, so `applicable=True`; what
    # does not apply is the BAND, because `_DIMENSIONS` gives it `lower_is_better: None` and a
    # dimension with no better/worse direction cannot have a wall. Marking it `applicable=False`
    # would say the box does not have it, which is false and hides a real reading.
    threads = evidence.get("threads_launched")
    out.append(DimensionRecord(
        dimension_id="threads_launched",
        measured=float(threads) if isinstance(threads, (int, float)) else None,
        ceiling=None,
        ceiling_provenance="reported as a quantity; this dimension has no better/worse direction",
        # `slack` would be the tempting default and it is the wrong one: it implies headroom in a
        # direction that does not exist.
        verdict="unknown" if threads is None else "not-applicable",
        applicable=True,
        not_applicable_reason=("" if threads is None else
                               "no polarity: more threads is neither better nor worse, so no band "
                               "applies to this reading"),
        higher_is_better=False, unit=str(_DIMENSIONS["threads_launched"]["unit"])))

    # NOTE on `num_warps`, which the plan's vocabulary lists: it is NOT in `classify()`'s evidence
    # (grep: 0 occurrences) and not in `conversion._DIMENSIONS` either, so there is nothing to read.
    # Deliberately not given a record built from a default -- a dimension fabricated from a missing
    # key is exactly the "confident constant" failure mode, and it would also inflate J2-9's count
    # of emitted dimensions without measuring anything.

    return DimensionState(candidate_id=candidate_id,
                          structural_signature=structural_signature,
                          records=tuple(out))


def unreachable_ceilings(evidence: dict[str, Any]) -> tuple[str, ...]:
    """Roofs that exist on this box but are NOT the roof this candidate is against (J2-4).

    Rule 4 of §1.5.3: a ceiling's reachability is derived, never assumed. Roofline's own paper
    overturns its ceiling order on SpMV and defines the upper ceilings to include work "inherently
    lacking in a kernel", and that is the L3:48 accident -- the framework kept aiming at a
    tensor-core roof the task could not use, 8 of 8 tensor-core candidates were rejected, and the
    winning 2.09 ms / 8.90x used none.

    Only states what is MEASURED in `evidence`. Three cases, in decreasing directness:

      * `backend_reachable_frac` (G10). The nominal roof is the max over measured paths, and Triton
        -- the backend every candidate here is written in -- reaches only part of it at some
        precisions. This is the case where "you are at 84% of the roof" and "84% is all your backend
        can reach" call for OPPOSITE actions, and they are indistinguishable without the number.
      * `dram_applicable=False` (G26). The working set fits in L2, so the DRAM roof is not the wall.
      * which compute roof is IN FORCE. When the instruction mix contains no tensor-core op, the
        percentage is against fp32 and the tensor-core roof is not the denominator -- so a
        recommendation aimed at it would be aimed at a number nothing here was measured against.

    NOT covered here, deliberately: "this TASK cannot reach the tensor-core roof at all", which on
    L3:48 was established by 8 of 8 tensor-core candidates failing CORRECTNESS. That is
    cross-candidate history, and it is in neither `evidence` nor this candidate's own measurements.
    Inventing it from a single candidate's instruction mix would be the same error as the nominal
    roof itself: a confident statement with nothing behind it. It belongs where the ceiling's
    provenance is built (S3), not here.
    """
    out: list[str] = []

    reach = evidence.get("backend_reachable_frac")
    if isinstance(reach, (int, float)) and reach < 1.0:
        ceiling = evidence.get("compute_ceiling_used") or "compute"
        out.append(
            "the %s roof is only %.1f%% reachable from Triton on this box, and every candidate here "
            "is Triton. The remaining %.1f%% is a code-generator gap, not a tiling gap: closing it "
            "would mean changing backend, so a recommendation that aims at the full roof is aiming "
            "past what this backend can do." % (ceiling, float(reach) * 100.0,
                                                (1.0 - float(reach)) * 100.0))

    if evidence.get("dram_applicable") is False:
        reason = evidence.get("dram_inapplicable_reason") or (
            "the working set fits in L2, so DRAM is not the path the traffic takes")
        out.append("the DRAM roof does not apply to this candidate: %s" % reason)

    uses_tc = evidence.get("uses_tensor_cores")
    ceiling_used = evidence.get("compute_ceiling_used")
    if uses_tc is False and ceiling_used:
        out.append(
            "the tensor-core roof is NOT the denominator for this candidate: its compiled "
            "instruction mix contains no tensor-core operation, so the compute percentage above is "
            "against the %s roof. Whether this task could use tensor cores at all is a separate "
            "question this measurement does not answer." % ceiling_used)

    return tuple(out)
