"""S3: where a ceiling came from, and how much room a dimension has left.

TWO GAPS, and they are different kinds of thing.

(1) PROVENANCE. `ceiling_provenance` was 0 occurrences repo-wide before S2, and what S2 added was a
free-form string. That is the same unfalsifiable shape S2d just replaced on the expectation side: a
sentence cannot be checked, so a wrong denominator stays invisible. The positive control is an
incident, not a hypothesis -- an fp16 kernel scored against a tf32 ceiling read **107.8% of peak**
when its real figure was about 54-60%, and 107.8% reads downstream as "saturated, stop optimizing".
The instruction was completely inverted. A structured provenance can be ASKED whether its precision
matches the kernel's, which is the one question that would have caught it.

(2) LOWER BOUNDS. A ceiling answers "how far from the roof"; it does not answer "how much room is
left", because a dimension's floor is usually not zero. Only a measured or definitional floor counts
here. THREE dimensions have one and the rest do not, and reporting `unknown` for the rest is the
whole point:

    candidate_aten_bytes   compulsory_bytes from task_cost.py -- distinct inputs + parameters +
                           outputs, each counted once. Fusion can remove every intermediate and
                           cannot remove these, so this is a REAL floor, measured on the reference.
    candidate_aten_ops     1. A fully fused implementation dispatches one op. Definitional.
    n_spills               0. Definitional; a spill is never required.

    n_regs                 UNKNOWN. Not derivable: 13 non-monotone slices were measured, with BK
                           16->32 dropping 58 registers while 32->64 added 87 and hit the 255 cap.
                           Even the SIGN is unreliable, so a floor would be a guess.
    shared_bytes           UNKNOWN. No closed form -- a candidate formula matched 0 of 96 measured
                           configurations, and the three points that motivated it were one family's
                           coincidence.
    occupancy              UNKNOWN. It is an OUTPUT of the register/shared/warp budget, not an
                           independent quantity, so its floor is whatever those allow. Deriving one
                           would mean deriving the register floor, which is the line above.
    peak_alloc_bytes       compulsory_bytes, weakly: the allocator must hold the distinct inputs and
                           outputs at once. Reported with reduced confidence because the allocator's
                           own caching sits between the two, so it is a bound on a different
                           quantity than the one measured.
    threads_launched       NOT APPLICABLE. No polarity, so "room left" has no direction to be in.

WHY `unknown` MUST BE REACHABLE, stated as an acceptance criterion in its own right: an invented
lower bound is worse than no lower bound. A fabricated floor makes a dimension look nearly exhausted
and moves effort away from it, and nothing downstream can tell a derived floor from a measured one
unless the record says which. So `LowerBound.source` is a closed vocabulary and `"none"` is a legal,
common value.

A MEASUREMENT BELOW ITS OWN FLOOR IS REPORTED, NEVER CLAMPED. This is not hypothetical here:
`candidate_aten_bytes` is itself a LOWER bound on the candidate's real traffic (a fused kernel does
its work inside one aten call and the intermediate bytes are never dispatched), so a well-fused
candidate can read BELOW compulsory_bytes. Clamping would say "0% room left, you are at the floor",
which is a false statement about a candidate that simply is not measurable this way. Same discipline
as `check_reading`'s out-of-range rule, and for the same reason: a clamp preserves the wrong action.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Where a number came from. Closed, because the whole point is that a reader can tell a measurement
# from a definition from a guess -- and there is no member for "guess".
CeilingSource = Literal[
    "measured",           # measured on THIS box by run_calibrate
    "device_query",       # the hardware's own declared limit (registers/thread, shared/block, VRAM)
    "definitional",       # true by definition (a spill floor of zero; occupancy's 1.0 roof)
    "task_lower_bound",   # measured on the REFERENCE: task_cost's compulsory traffic
    "none",               # no ceiling for this dimension on this box. A legal, common state.
]

BoundSource = Literal["task_lower_bound", "definitional", "none"]


class CeilingProvenance(BaseModel):
    """Which denominator a fraction is against, in fields rather than in a sentence.

    `precision` and `backend` are separate fields specifically so `precision_mismatch()` can exist.
    Folding them into a note would leave the 107.8% incident undetectable again -- the note would say
    "tensor-core ceiling" and nothing could ask "of WHICH precision".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: CeilingSource = "none"
    # The precision the ceiling was measured AT. Empty when the dimension has no precision (a
    # register count does not), which is different from "unknown precision".
    precision: str = ""
    # `cublas`, `triton`, or empty. Load-bearing rather than descriptive: on box 1 the two disagree in
    # BOTH directions (Triton reaches 84.1% of cuBLAS at fp32 but 109.6% at fp16), so "84% of the
    # roof" and "84% is all your backend can reach" are different statements needing opposite actions,
    # and they are indistinguishable without this field.
    backend: str = ""
    # When, and on what. A ceiling measured under a different Triton generates different code, so a
    # figure from another identity does not describe this box (G32).
    measured_at: str = ""
    calibration_identity: str = ""
    note: str = ""

    def precision_mismatch(self, candidate_precision: str | None) -> str | None:
        """The 107.8% control, as a question the record can answer.

        Returns a sentence when the kernel computes in one precision and the denominator was measured
        at another, else None. Silent when either side is unknown -- an absent precision is not
        evidence of a mismatch, and claiming one would be the same error in the other direction.
        """
        if not candidate_precision or not self.precision:
            return None
        if candidate_precision == self.precision:
            return None
        return (
            "PRECISION MISMATCH: this kernel computes in %s and the ceiling it is measured against "
            "was taken at %s. The percentage is therefore against the wrong denominator, and the "
            "error is not small -- an fp16 kernel scored against a tf32 ceiling read 107.8%% of peak "
            "when its real figure was about 54-60%%, which downstream reads as 'saturated, stop "
            "optimizing' for a kernel with most of its headroom left."
            % (candidate_precision, self.precision))


class LowerBound(BaseModel):
    """How much room a dimension has left, or an explicit statement that it is not derivable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # None means UNKNOWN, and `source == "none"` says so in words. Not 0.0: a floor of zero is a
    # strong claim (it says the dimension can be eliminated entirely) and is true only for spills.
    floor: float | None = None
    source: BoundSource = "none"
    # Why there is no floor, when there is none. Required in that case for the same reason
    # `not_applicable_reason` is required on a DimensionRecord: "not derivable" and "nobody derived
    # it" lead to different actions.
    reason: str = ""
    # `measured - floor`, when both exist. Named "room" rather than "headroom" because headroom in
    # this project already means distance to a CEILING, and this is distance to a FLOOR.
    room: float | None = None
    room_frac: float | None = None
    # True when the reading is BELOW its own floor, which is possible and is never clamped -- see the
    # module docstring. A flag rather than a correction, so the reader sees the real state.
    below_floor: bool = False
    confidence: float = 1.0


# Floors that hold by definition, independent of task and box.
_DEFINITIONAL_FLOORS: dict[str, tuple[float, str]] = {
    "n_spills": (0.0, "a spill is never required: zero spills is always attainable in principle, so "
                      "any spill at all is room to recover"),
    "candidate_aten_ops": (1.0, "a fully fused implementation dispatches ONE aten op, so one is the "
                                "floor and everything above it is unfused work"),
}

# Dimensions with no derivable floor, and the measurement that says so. Kept as data rather than as
# prose in a branch, so the reason travels with the record and cannot be dropped by a later edit.
_NOT_DERIVABLE: dict[str, str] = {
    "n_regs": ("not derivable: register demand has no closed form in the knobs, and the SIGN is "
               "unreliable -- 13 non-monotone slices were measured, with BK 16->32 dropping 58 "
               "registers while 32->64 added 87 and hit the 255 cap"),
    "shared_bytes": ("not derivable: a candidate closed-form formula matched 0 of 96 measured "
                     "configurations, and the three points that appeared to confirm it were one "
                     "family's coincidence"),
    "occupancy": ("not derivable: occupancy is an OUTPUT of the register, shared-memory and warp "
                  "budgets rather than an independent quantity, so its floor is whatever those "
                  "allow -- deriving one would mean deriving the register floor, which is not "
                  "derivable either"),
    "threads_launched": ("no polarity: this dimension has no better/worse direction, so 'room left' "
                         "has no direction to be in"),
}


def _room(measured: float | None, floor: float | None) -> tuple[float | None, float | None, bool]:
    if measured is None or floor is None:
        return None, None, False
    room = measured - floor
    frac = (room / measured) if measured else None
    return room, (round(frac, 4) if frac is not None else None), room < 0


def lower_bound(dimension_id: str, measured: float | None,
                task_cost: Any | None = None) -> LowerBound:
    """This dimension's floor on this task, or an explicit `unknown`.

    `task_cost` is a `TaskCost` (read by attribute so a replayed dict shim works). It is the ONLY
    source of a task-specific floor, and it is deliberately measured on the REFERENCE: a per-candidate
    count would measure what that candidate happens to do, which is the quantity being optimized and
    therefore useless as a yardstick. `task_cost.py` is not modified by this module -- S3 reads it.
    """
    if dimension_id in _DEFINITIONAL_FLOORS:
        floor, why = _DEFINITIONAL_FLOORS[dimension_id]
        room, frac, below = _room(measured, floor)
        return LowerBound(floor=floor, source="definitional", reason=why,
                          room=room, room_frac=frac, below_floor=below)

    compulsory = getattr(task_cost, "compulsory_bytes", None) if task_cost is not None else None

    if dimension_id == "candidate_aten_bytes":
        if not compulsory:
            return LowerBound(source="none", reason=(
                "the task's compulsory traffic was not measured on this run, so there is no floor to "
                "compare against -- not that the floor is zero"))
        room, frac, below = _room(measured, float(compulsory))
        return LowerBound(
            floor=float(compulsory), source="task_lower_bound",
            reason=("distinct inputs, parameters and outputs, each counted once, measured on the "
                    "REFERENCE. Fusion can remove every intermediate and cannot remove these"),
            room=room, room_frac=frac, below_floor=below)

    if dimension_id == "peak_alloc_bytes":
        if not compulsory:
            return LowerBound(source="none", reason=(
                "the task's compulsory traffic was not measured, so the weak allocation floor it "
                "would give is unavailable"))
        room, frac, below = _room(measured, float(compulsory))
        return LowerBound(
            floor=float(compulsory), source="task_lower_bound",
            reason=("a WEAK floor: the allocator must hold the task's distinct inputs and outputs at "
                    "once. Weak because the caching allocator sits between the two, so this bounds a "
                    "different quantity than the one measured"),
            room=room, room_frac=frac, below_floor=below,
            # Reduced on purpose: a bound on a neighbouring quantity is not a bound on this one, and
            # a bare 1.0 here would invite reading the room figure as exact.
            confidence=0.5)

    if dimension_id in _NOT_DERIVABLE:
        return LowerBound(source="none", reason=_NOT_DERIVABLE[dimension_id])

    return LowerBound(source="none", reason=(
        "no floor is defined for this dimension, and none is derived here -- deriving one would be "
        "inventing it"))


def describe(dimension_id: str, bound: LowerBound) -> str:
    """One line for a prompt or a report. Says `unknown` out loud when it is unknown."""
    if bound.floor is None:
        return "%s — room left UNKNOWN: %s" % (dimension_id, bound.reason)
    if bound.below_floor:
        return (
            "%s — reads BELOW its own floor (%s against a floor of %s). Not an error and not "
            "clamped: for aten traffic this is the expected reading from a well-fused candidate, "
            "whose intermediate bytes are never dispatched and so are never counted. It means this "
            "dimension cannot bound this candidate, NOT that the candidate is at the floor."
            % (dimension_id, _num(bound.floor + (bound.room or 0.0)), _num(bound.floor)))
    pct = "" if bound.room_frac is None else " (%.0f%% of the current reading)" % (
        bound.room_frac * 100.0)
    return "%s — room left: %s above a floor of %s%s [%s]" % (
        dimension_id, _num(bound.room), _num(bound.floor), pct, bound.source)


def _num(v: float | None) -> str:
    if v is None:
        return "?"
    if abs(v) >= 1e6:
        return "%.3g" % v
    return "%g" % round(v, 3)
