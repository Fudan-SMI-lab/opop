"""Does the median fallback actually reproduce the PRE-FIX rule, or is it more permissive?

`fix-boundary-direction-follows-the-winning-trial.md` describes the shipped second pass as
falling back to "the median's aim" -- i.e. the behaviour that existed before the fix. That claim
is worth checking, because the two are implemented in different places:

  * the pre-fix flag came from `_param_stat` (stats.py), which requires THREE things: the
    anchor sits on the measured edge, the median curve's 3-value tail is MONOTONE toward that
    edge, and the range beyond it is either absent or entirely failing;
  * the fallback `_median_direction` (orchestrator.py) checks only the FIRST of those.

If the fallback omits the monotone and beyond tests, it is not a restoration of the old rule; it
is a wider rule that can request knobs neither the old nor the new rule would have asked for --
which would make the "floor, not veto" framing wrong in the permissive direction.

Replays every space on disk and reports, per knob, what each of the three rules says.
Read-only; safe against a live run.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.models.core import ParameterSpace  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402


def old_rule(ps, sp) -> tuple[bool, str | None]:
    """`_param_stat`'s edge test, anchored on the MEDIAN's pick -- the pre-fix behaviour."""
    lat = ps.latency_by_value or {}
    if len(lat) < 2 or sp is None:
        return False, None
    try:
        choices = sp.domain(ps.name).choices
    except KeyError:
        return False, None
    measured = [c for c in choices if repr(c) in lat]
    if len(measured) < 2:
        return False, None
    anchor = min(measured, key=lambda c: lat[repr(c)])
    idx = measured.index(anchor)
    at_min, at_max = idx == 0, idx == len(measured) - 1
    if not (at_min or at_max):
        return False, None
    full = choices.index(anchor)
    beyond = choices[:full] if at_min else choices[full + 1:]
    fails = ps.failure_rate_by_value or {}
    beyond_blocked = all(
        fails.get(repr(c), 0.0) >= 1.0 or repr(c) not in lat for c in beyond
    )
    ordered = [lat[repr(c)] for c in measured]
    tail = ordered[:3] if at_min else ordered[-3:]
    monotone = len(tail) >= 2 and (
        all(a <= b for a, b in zip(tail, tail[1:])) if at_min
        else all(a >= b for a, b in zip(tail, tail[1:]))
    )
    if monotone and (not beyond or beyond_blocked):
        return True, ("min" if at_min else "max")
    return False, None


def fallback_rule(ps, sp) -> tuple[bool, str | None]:
    """`_median_direction` as shipped: argmin-at-edge only."""
    lat = ps.latency_by_value or {}
    if len(lat) < 2 or sp is None:
        return False, None
    try:
        choices = sp.domain(ps.name).choices
    except KeyError:
        return False, None
    measured = [c for c in choices if repr(c) in lat]
    if len(measured) < 2:
        return False, None
    best = min(measured, key=lambda c: lat[repr(c)])
    if best == measured[0]:
        return True, "min"
    if best == measured[-1]:
        return True, "max"
    return False, None


runs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/run-l3-*"))
tally = {"same": 0, "fallback_wider": 0, "fallback_narrower": 0}
examples: list[str] = []
for run in runs:
    ev = Path(run) / "events.jsonl"
    if not ev.exists():
        continue
    spaces: dict[str, ParameterSpace] = {}
    for line in open(ev, encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "SPACE_PUBLISHED":
            sp = ParameterSpace.model_validate(e["payload"]["space"])
            spaces[sp.space_id] = sp
        elif e["type"] == "STATS_DONE":
            st = TuningStats.model_validate(e["payload"]["stats"])
            sp = spaces.get(st.space_id)
            for ps in st.param_stats:
                o_flag, o_dir = old_rule(ps, sp)
                f_flag, f_dir = fallback_rule(ps, sp)
                if (o_flag, o_dir) == (f_flag, f_dir):
                    tally["same"] += 1
                elif f_flag and not o_flag:
                    tally["fallback_wider"] += 1
                    if len(examples) < 12:
                        examples.append(
                            f"{Path(run).name} {st.space_id} {ps.name}: "
                            f"old=None fallback={f_dir} "
                            f"median={ps.best_value!r} winner={ps.best_trial_value!r}"
                        )
                else:
                    tally["fallback_narrower"] += 1

print("Per-knob agreement between the PRE-FIX rule and the shipped median fallback:")
for k, v in tally.items():
    print(f"  {k:20s} {v}")
print("\nKnobs the fallback requests that the pre-fix rule would NOT have:")
for line in examples:
    print("  " + line)
