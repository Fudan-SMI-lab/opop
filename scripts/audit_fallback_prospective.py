"""What would each rule request for a space that has NOT been expanded yet?

`fix-boundary-direction-follows-the-winning-trial.md` measured the anchored pass and the median
fallback retrospectively, then confirmed the anchored pass on `run-l3-43`'s first expansion (the
easy case: median and winner agreed). What it has never seen live is the case the FALLBACK
exists for -- a space where the anchored pass returns nothing at all, so an expansion would be
cancelled without it.

`sp-cc814089` (cand-a988ff79, L3:43) is that case waiting to happen: 11 knobs, 6 of them on a
median edge, only 2 on the winner's edge, and every one reported `at_boundary=False`. It has
tuned once and not yet been expanded, so this prints the prediction BEFORE the run acts --
which makes the subsequent SPACE_EXPANDED event a real prospective test rather than a
post-hoc reading.

Prints, for the newest L3:43 run and for every space that has stats:
  - what the anchored pass requests (the shipped first pass),
  - what the median fallback would request (the shipped second pass),
  - whether the fallback is what saves the expansion from being cancelled.

Read-only over events.jsonl; runs safely against a live run.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand  # noqa: E402
from kernel_optimizer.models.core import ParameterSpace  # noqa: E402
from kernel_optimizer.models.reports import TuningStats  # noqa: E402

pattern = sys.argv[1] if len(sys.argv) > 1 else "runs/run-l3-43-20260906-*"
run = sorted(glob.glob(pattern))[-1]
print(f"run: {run}\n")

evs = [json.loads(l) for l in open(Path(run) / "events.jsonl", encoding="utf-8")]

spaces: dict[str, ParameterSpace] = {}
stats_by_space: dict[str, TuningStats] = {}
expanded: dict[str, list[dict]] = {}
for e in evs:
    if e["type"] == "SPACE_PUBLISHED":
        sp = ParameterSpace.model_validate(e["payload"]["space"])
        spaces[sp.space_id] = sp
    elif e["type"] == "STATS_DONE":
        st = TuningStats.model_validate(e["payload"]["stats"])
        stats_by_space[st.space_id] = st
    elif e["type"] == "SPACE_EXPANDED":
        expanded.setdefault(e["payload"].get("candidate_id") or "?", []).append(
            e["payload"].get("knobs") or []
        )

for sid, st in stats_by_space.items():
    sp = spaces.get(sid)
    # idle_frac=1.0 isolates the direction logic from the resource-headroom precondition,
    # which is a separate gate and not what this probe is about.
    anchored = boundary_knobs_to_expand(st, idle_frac=1.0, space=sp)
    print(f"== {sid}  ({len(st.param_stats)} knobs)")
    print(f"   shipped request : {[(k['name'], k['direction']) for k in anchored]}")
    if not anchored:
        print("   -> the anchored pass is EMPTY; without the median fallback this")
        print("      expansion would be CANCELLED and its fresh trial budget forfeited.")
    on_median_edge = []
    for ps in st.param_stats:
        lat = ps.latency_by_value or {}
        if not sp or len(lat) < 2:
            continue
        try:
            choices = sp.domain(ps.name).choices
        except KeyError:
            continue
        meas = [c for c in choices if repr(c) in lat]
        if len(meas) < 2:
            continue
        best = min(meas, key=lambda c: lat[repr(c)])
        if best == meas[0]:
            on_median_edge.append((ps.name, "min", ps.best_trial_value))
        elif best == meas[-1]:
            on_median_edge.append((ps.name, "max", ps.best_trial_value))
    print(f"   median edges    : {on_median_edge}")
    print()

print("SPACE_EXPANDED so far:", json.dumps(expanded))
