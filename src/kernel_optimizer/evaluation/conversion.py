"""Did a resource change CONVERT INTO SPEED? (G3)

The harness measures resources -- registers, shared memory, occupancy, peak memory, traffic --
and it measures latency. Nothing tied the two together, so a rewrite that improved every resource
figure while latency stayed put was recorded exactly like one that improved both.

That is not a hypothetical. Across 23 measured rewrites with a gain above the noise floor, 11
relieved no binding resource dimension at all and still got faster; the reverse case -- resources
better, latency flat -- is therefore certain to occur too, and had no way to be reported.

So the rule this module enforces: LATENCY IS THE ONLY FINAL CRITERION. A resource change is a
means, never an end. Every resource delta is reported together with what it bought, and a delta
that bought nothing is labelled `no_conversion` rather than being quietly averaged into the
record as a success.

The third category is the informative one. A resource that improved substantially while latency
did not move is direct evidence that THAT RESOURCE WAS NOT THE LIMIT -- which is a more reliable
statement than any utilisation label, because it is a controlled comparison between two real
measurements of the same family rather than a ratio against an assumed ceiling.

WHAT THIS DELIBERATELY DOES NOT DO. It does not rank dimensions against each other, and it does
not compute a composite efficiency number. Pressure deltas have no common unit across dimensions
-- shared_bytes/101376 moves by tens of thousands whenever a tile changes while n_regs/255 moves
by a few, measured median swings 0.162 vs 0.012, a 13.5x difference that comes entirely from the
normalising denominator. So each dimension is reported with its own absolute delta and its own
verdict, and no argmax is taken across them.
"""
from __future__ import annotations

from typing import Any

# Resource dimensions compared before/after a rewrite, with the direction that counts as an
# IMPROVEMENT for each. Explicit per dimension, because getting this wrong does not fail loudly:
# applying "lower is better" to occupancy inverts every verdict and still produces a clean table.
# (That exact error once reported 0 multi-binding candidates while the motivating case sat in the
# input.)
_DIMENSIONS: dict[str, dict[str, Any]] = {
    "n_regs": {"lower_is_better": True, "unit": "registers/thread"},
    "n_spills": {"lower_is_better": True, "unit": "local-memory slots"},
    "shared_bytes": {"lower_is_better": True, "unit": "bytes"},
    "peak_alloc_bytes": {"lower_is_better": True, "unit": "bytes"},
    "candidate_aten_bytes": {"lower_is_better": True, "unit": "bytes (lower bound)"},
    "candidate_aten_ops": {"lower_is_better": True, "unit": "aten ops"},
    "threads_launched": {"lower_is_better": None, "unit": "threads"},
    # occupancy lives in a sub-dict on the profile and HIGHER is better, unlike every other
    # entry here -- the one dimension whose polarity is inverted.
    "occupancy": {"lower_is_better": False, "unit": "fraction", "nested": True},
}

# A resource has to move by more than this fraction of its own previous value before the change is
# called real. Not a physical quantity: it exists so that a 1-register or 128-byte difference is
# not reported as a resource change with a latency verdict attached to it.
_MATERIAL_RESOURCE_DELTA = 0.05


def _read(profile: Any, name: str) -> float | None:
    """One dimension's value off a ProfileRecord, or None when it was not measured.

    None must never be coerced to 0. An unmeasured shared_bytes read as 0 would make every rewrite
    look like it eliminated all shared memory.
    """
    if profile is None:
        return None
    if _DIMENSIONS.get(name, {}).get("nested"):
        occ = getattr(profile, "occupancy", None)
        if isinstance(occ, dict):
            v = occ.get("occupancy")
            return float(v) if isinstance(v, (int, float)) else None
        return None
    v = getattr(profile, name, None)
    return float(v) if isinstance(v, (int, float)) else None


def conversion_verdict(ms_before: float | None, ms_after: float | None,
                       profile_before: Any, profile_after: Any,
                       min_improvement_pct: float) -> dict:
    """What each resource change bought, plus the round's overall conversion verdict.

    Returns a dict merged into FAMILY_ROUND_RECORDED, so the judgement is in the event log rather
    than recomputed later from partial data.

    `conversion` is one of:
      improved          latency fell by more than the noise floor
      no_conversion     a resource improved materially and latency did NOT move -- the informative
                        case: that resource was not the limit
      regressed         latency rose
      flat              nothing moved materially, in either latency or resources
      unknown           latency could not be compared
    """
    out: dict = {}
    if not (isinstance(ms_before, (int, float)) and isinstance(ms_after, (int, float))
            and ms_before > 0):
        return {"conversion": "unknown",
                "conversion_note": "latency before/after not both available"}

    gain_pct = 100.0 * (ms_before - ms_after) / ms_before
    out["latency_gain_pct"] = round(gain_pct, 3)
    out["latency_ms_before"] = round(float(ms_before), 4)
    out["latency_ms_after"] = round(float(ms_after), 4)

    deltas: dict[str, dict] = {}
    improved_resources: list[str] = []
    for name, spec in _DIMENSIONS.items():
        b, a = _read(profile_before, name), _read(profile_after, name)
        if b is None or a is None:
            continue
        d = a - b
        rel = (abs(d) / abs(b)) if b else (1.0 if d else 0.0)
        material = rel > _MATERIAL_RESOURCE_DELTA
        lower_better = spec["lower_is_better"]
        if lower_better is None:
            direction = "changed" if material else "flat"
        elif d == 0:
            direction = "flat"
        else:
            better = (d < 0) if lower_better else (d > 0)
            direction = ("improved" if better else "worsened") if material else "flat"
        deltas[name] = {"before": b, "after": a, "delta": round(d, 4),
                        "rel": round(rel, 4), "unit": spec["unit"],
                        "direction": direction}
        if direction == "improved":
            improved_resources.append(name)
    if deltas:
        out["resource_deltas"] = deltas
    out["resources_improved"] = improved_resources

    # Latency decides. This ordering is the whole point of the module: a resource improvement can
    # never make a round a success on its own.
    if gain_pct >= min_improvement_pct:
        out["conversion"] = "improved"
        out["conversion_note"] = (
            "latency fell %.2f%%; resource changes that accompanied it: %s"
            % (gain_pct, ", ".join(improved_resources) or "none measured as improved"))
    elif gain_pct <= -min_improvement_pct:
        out["conversion"] = "regressed"
        out["conversion_note"] = "latency rose %.2f%%" % (-gain_pct)
    elif improved_resources:
        out["conversion"] = "no_conversion"
        out["conversion_note"] = (
            "%s improved but latency moved only %.2f%% (below the %.1f%% floor), so those "
            "resources were NOT the limit for this structure. This is evidence about where the "
            "limit is not, and it is the reason a resource improvement is never scored as a "
            "success on its own."
            % (", ".join(improved_resources), gain_pct, min_improvement_pct))
    else:
        out["conversion"] = "flat"
        out["conversion_note"] = ("neither latency nor any measured resource moved materially "
                                  "(latency %.2f%%)" % gain_pct)
    return out
