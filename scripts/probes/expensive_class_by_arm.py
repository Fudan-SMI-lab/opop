"""Is the expensive configuration class hitting one arm more than the other, and does that explain the rate gap?

WHY. The treatment arm has now stalled twice on the same shape: NUM_WARPS=1 with a large tile and ieee.
The first cost 1081 s (39.9% of that arm's whole GPU time); a second was live at 6+ minutes when this was
written. Twice is not a coincidence worth ignoring, and "one outlier" is a different claim from "this arm
draws the expensive class more often" -- the first is variance, the second would be a systematic rate
difference that compounds over 12 h.

WHAT DECIDES IT. The class is a property of the CONFIGURATION, and the two arms tune DIFFERENT candidates
with different spaces, so the relevant quantity is not "how many did each arm run" alone but "how often
does the class appear among the configs each arm's space makes available and TPE actually draws". A count
alone would confound the class's cost with the space's shape.

So this reports, per arm and per space: how many trials fall in the class, what they cost against the
arm's own median, and -- the part that separates variance from a systematic difference -- whether the
class is even REACHABLE in that space (does the space declare NUM_WARPS=1 at all?). An arm whose spaces
do not offer NUM_WARPS=1 cannot draw it, and that is a fact about the candidate the generator produced,
not about the switch under test.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/expensive_class_by_arm.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")

# The class, defined by what was measured rather than guessed: NUM_WARPS=1 starves the compiler of
# parallelism so ptxas works a single enormous scheduling region, and `ieee` forbids the tensor-core
# path, so the tile is realised as scalar FMAs. Large tiles multiply both.
def _in_class(v: dict) -> bool:
    nw = v.get("NUM_WARPS")
    if nw is None or int(nw) > 1:
        return False
    tile = max((int(v.get(k) or 0) for k in ("BLOCK_M", "BLOCK_N", "BM_QKV", "BM_PROJ")), default=0)
    return tile >= 128


def report(arm: str) -> dict:
    runs = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    if not runs:
        print(f"=== {arm}: no run")
        return {}
    run = runs[-1]
    spaces: dict[str, dict] = {}
    per_space: dict[str, list[tuple[float, bool, str]]] = {}
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t, p = e.get("type"), (e.get("payload") or {})
        if t == "SPACE_PUBLISHED":
            sp = p.get("space") or {}
            doms = {d["name"]: list(d["choices"]) for d in (sp.get("domains") or [])}
            sid = str(sp.get("space_id") or "")
            if sid:
                spaces[sid] = doms
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or {}
            v = (tr.get("params") or {}).get("values") or {}
            w = tr.get("job_wall_s")
            if not isinstance(w, (int, float)):
                continue
            per_space.setdefault(str(tr.get("space_id")), []).append(
                (float(w), _in_class(v), str(tr.get("failure_kind") or "ok")))

    print(f"=== {arm}  {run.name}")
    all_walls = [w for rows in per_space.values() for w, _, _ in rows]
    med = statistics.median(all_walls) if all_walls else 0.0
    tot_class = tot_class_s = 0
    for sid, rows in per_space.items():
        doms = spaces.get(sid) or {}
        nw = doms.get("NUM_WARPS") or doms.get("NUM_WARPS_QKV") or []
        reachable = 1 in [int(x) for x in nw if isinstance(x, (int, float))]
        hits = [(w, fk) for w, c, fk in rows if c]
        tot_class += len(hits)
        tot_class_s += sum(w for w, _ in hits)
        print("    %-14s trials=%-4d class=%-3d class_seconds=%-7.0f NUM_WARPS=1 declared=%s" % (
            sid, len(rows), len(hits), sum(w for w, _ in hits), reachable))
        for w, fk in sorted(hits, reverse=True)[:3]:
            print("        %6.0f s  %-26s (%.1fx this arm's median %.0f s)" % (
                w, fk, w / med if med else 0, med))
    tot_s = sum(all_walls)
    print("    class total: %d trials, %.0f s = %.1f%% of this arm's %.0f s of GPU time" % (
        tot_class, tot_class_s, 100.0 * tot_class_s / tot_s if tot_s else 0, tot_s))
    return {"n": len(all_walls), "class_n": tot_class, "class_s": tot_class_s,
            "total_s": tot_s, "median": med}


def main() -> int:
    t = report("s7-treatment")
    print()
    c = report("s7-control")
    print()
    print("=" * 84)
    if t and c:
        print("  class share of GPU time:  treatment %.1f%%   control %.1f%%" % (
            100.0 * t["class_s"] / t["total_s"] if t["total_s"] else 0,
            100.0 * c["class_s"] / c["total_s"] if c["total_s"] else 0))
        print("  class share of trials:    treatment %.1f%%   control %.1f%%" % (
            100.0 * t["class_n"] / t["n"] if t["n"] else 0,
            100.0 * c["class_n"] / c["n"] if c["n"] else 0))
        print()
        print("  HOW TO READ IT. If the class is UNREACHABLE in an arm's spaces (NUM_WARPS=1 not")
        print("  declared), that arm cannot draw it and the asymmetry is a property of the CANDIDATE the")
        print("  generator produced, not of the switch. If it is reachable in both and one arm draws it")
        print("  far more, that is TPE's sampling and still not the switch -- S7 enqueued 0 points, so")
        print("  it cannot have steered anything. Either way the cost belongs in the parity note, and")
        print("  `check_arm_search_parity.py` already prices it via the MEDIAN gap rather than the sum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
