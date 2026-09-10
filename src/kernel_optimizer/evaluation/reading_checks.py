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
