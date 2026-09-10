"""Self-checks on collected resource readings (G13).

WHY THIS EXISTS. Withdrawing the TMA-closure design removed a free consistency check: components
had to sum to a declared total with a non-negative residual, so any arithmetic mistake was caught
by construction. Nothing replaced it, and the S2 acceptance criteria require a substitute.

The two checks here are not invented. Each is the generalisation of a real failure that produced a
plausible, wrong table rather than an error:

  CONSTANT ACROSS CONFIGURATIONS. A probe read `n_regs` as 56 for all 108 configurations of a
  sweep. Every downstream consumer accepted it: 56 is a perfectly ordinary register count, no
  field was None, no exception was raised. The accessor was handing back the same cached kernel
  every time. So a "not None" check is not enough -- the dangerous failure is a CREDIBLE CONSTANT,
  and it is only visible across a set of readings that should differ.

  OUT-OF-RANGE. A fraction above 1.0 is physically impossible and means the denominator does not
  apply, but it does not raise: an fp16 kernel scored against a tf32 ceiling read 107.8% of peak,
  which downstream reads as "saturated, stop optimizing" for a kernel with headroom left. Nsight's
  own documentation notes percent-of-peak metrics can exceed 100%, so this must be reported rather
  than clamped -- a clamp preserves the wrong action.

WHAT THIS IS NOT. Not a correctness gate. It never rejects a candidate or fails a run; it
annotates the collection so a reader can tell "measured" from "the collector was broken". A check
that could fail a run would make a diagnostic defect look like a candidate defect, which is the
inversion these fixes exist to prevent.
"""
from __future__ import annotations

from typing import Any

# Hard architectural limits and definitional ranges, per dimension. A value outside these did not
# come from the hardware, whatever produced it.
_RANGES: dict[str, tuple[float, float]] = {
    "n_regs": (0, 255),
    "n_spills": (0, 1 << 22),
    "shared_bytes": (0, 1 << 20),
    "num_warps": (1, 32),
    "num_stages": (1, 16),
    "occupancy": (0.0, 1.0),
    "peak_alloc_bytes": (0, 1 << 42),
    "candidate_aten_ops": (0, 1 << 20),
    "threads_launched": (0, 1 << 34),
}

# Dimensions that MUST vary across a set of structurally different configurations. Chosen because
# each is a compile-time consequence of the knobs a sweep varies: a set of different tile shapes
# that all report one register count has not been measured.
_SHOULD_VARY = ("n_regs", "shared_bytes")


def check_reading(values: dict[str, Any]) -> list[str]:
    """Range problems in one kernel's readings. Empty list means nothing detectable is wrong.

    Deliberately silent about missing values: None means "not collected on this path", which is a
    legitimate state that the notes fields already carry. Only a value that IS present and cannot
    be right is reported.
    """
    notes: list[str] = []
    for name, (lo, hi) in _RANGES.items():
        v = values.get(name)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if v < lo or v > hi:
            notes.append(
                "%s=%s is outside the possible range [%s, %s], so this reading did not come from "
                "the hardware and must not be used as a measurement" % (name, v, lo, hi))
    return notes


def check_collection(readings: list[dict[str, Any]]) -> list[str]:
    """Problems visible only ACROSS a set of readings that should differ.

    `readings` is one dict per configuration -- a resource-map sweep, or the kernels of one
    candidate compiled at different parameter values. Fewer than three is not enough to distinguish
    a broken accessor from a genuinely flat region, so the check declines rather than guessing: two
    equal readings are unremarkable, and calling them a defect would cry wolf on every small sweep.
    """
    notes: list[str] = []
    if len(readings) < 3:
        return notes
    for name in _SHOULD_VARY:
        vals = [r.get(name) for r in readings]
        present = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(present) < 3:
            continue
        if len(set(present)) == 1:
            notes.append(
                "%s read the identical value %s across all %d configurations. A credible constant "
                "is the characteristic signature of a broken accessor -- an earlier probe returned "
                "n_regs=56 for 108 different configurations because it re-read one cached kernel, "
                "and nothing downstream noticed because no field was None. Treat these readings as "
                "UNMEASURED until the accessor is re-checked."
                % (name, present[0], len(present)))
    return notes


def check_applicability(records: list[Any]) -> list[str]:
    """P4's missing half: is each record's `applicable` flag consistent with its own contents?

    The plan carried this as an open pre-condition ("`reading_checks.py` has no `applicable`
    consistency check, grep confirms 0 occurrences") and noted it structurally had to wait until S2
    defined the record shape. S2 defines it, so this is that check.

    `applicable` and `measured` answer different questions, and the whole reason for having both is
    that they stay distinguishable: a card without bf16 does not have that dimension, it does not
    have it "at zero". Same discipline `task_cost.py` already applies to `flop_count=0` -- legal for
    maxpool, a failure for matmul. Four inconsistencies are detectable from a record alone:

      1. applicable=False with a verdict other than not-applicable -- a judgement about a dimension
         that does not exist here.
      2. applicable=False with no reason -- indistinguishable from a collection failure, which is
         the exact confusion this field exists to prevent.
      3. applicable=True, measured present, a CEILING present, verdict `unknown` -- a band could
         have been assigned and was not, so something declined to judge without saying so. The
         ceiling clause matters: a measurement with no roof (aten traffic is a lower bound with
         nothing to divide by) genuinely cannot be banded, and `unknown` is honest there.
      4. applicable=True, measured absent, verdict anything but `unknown` -- a verdict with nothing
         behind it. `slack` is the dangerous one: it reads as headroom.
      5. verdict `not-applicable` with no reason, whatever `applicable` says. The two fields answer
         DIFFERENT questions and come apart on a real dimension: `threads_launched` exists and is
         measured (so it IS applicable) while no BAND applies to it, because it has no better/worse
         direction. Whichever of the two produced the verdict, the reason is the only thing that
         tells a reader which one it was.

    Reads attributes rather than importing the model, so this cannot create an import cycle and
    works on a replayed record shim as well as on the real class.
    """
    notes: list[str] = []
    for rec in records:
        dim = getattr(rec, "dimension_id", None) or "<unnamed>"
        applicable = getattr(rec, "applicable", None)
        measured = getattr(rec, "measured", None)
        verdict = getattr(rec, "verdict", None)
        reason = (getattr(rec, "not_applicable_reason", "") or "").strip()

        if verdict == "not-applicable" and not reason:
            notes.append(
                "%s reads verdict='not-applicable' with no reason. That verdict has two distinct "
                "causes -- the dimension does not exist on this box, or it exists and has no band "
                "(no better/worse direction) -- and without the reason a reader cannot tell which, "
                "so cannot tell whether the absent number is a gap or a property." % dim)

        if applicable is False:
            if verdict != "not-applicable":
                notes.append(
                    "%s is marked applicable=False but carries verdict=%r. A dimension that does "
                    "not exist on this box/candidate cannot also be judged; `slack` in particular "
                    "would read as headroom in a dimension that has none." % (dim, verdict))
            if not reason:
                notes.append(
                    "%s is marked applicable=False with no reason, which is indistinguishable from "
                    "a collection failure -- and telling those apart is the entire purpose of "
                    "having `applicable` as a field instead of using measured=0." % dim)
            continue

        if applicable is True:
            if measured is not None and verdict == "unknown":
                # A measurement with NO ceiling genuinely cannot be banded -- aten traffic is a
                # lower bound with no roof to divide by -- so `unknown` is the honest verdict there
                # and not a silent declination. Only flag the case where a ceiling exists.
                if getattr(rec, "ceiling", None) is not None:
                    notes.append(
                        "%s has a measured value (%s) and a ceiling but verdict='unknown': a band "
                        "could have been assigned and was not, so something declined to judge "
                        "without saying so." % (dim, measured))
            if measured is None and verdict not in ("unknown", None):
                notes.append(
                    "%s has verdict=%r with no measured value behind it. An unmeasured dimension "
                    "must read `unknown`; reporting a band states a conclusion the box never "
                    "supplied." % (dim, verdict))
    return notes
