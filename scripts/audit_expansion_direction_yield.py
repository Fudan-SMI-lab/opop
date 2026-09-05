"""Improvement K's yield, measured strictly and split by requested direction.

`audit_expansion_outcomes.py` answers "did the candidate's best latency improve
after the expansion?" -- which counts a win even when the improvement came from a
value that was already in the space (the re-tune got luckier). This script answers
the stricter question:

  did the post-expansion best config actually USE a value the expansion added?

and splits the answer by the direction `boundary_knobs_to_expand` requested, which
turns out to predict the outcome. Runs are dated against the hard-edge filter
commit, because the pre-filter NUM_WARPS=1 / NUM_STAGES=1 requests are already
fixed and pooling them understates the current behaviour.

Usage: python scripts/audit_expansion_direction_yield.py [runs_dir]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# The commit that stopped K expanding a knob whose range already touches a
# hardware floor (NUM_WARPS=1, NUM_STAGES=1).
HARD_EDGE_COMMIT = "4030458"


def commit_ts(rev: str) -> float | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct", rev],
                             capture_output=True, text=True, timeout=15,
                             cwd=str(Path(__file__).resolve().parents[1]))
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def expansions(run: Path):
    """Yield (post_fix_unknown_ts, direction, knob, added_values, best_used_added, gain_pct)."""
    p = run / "events.jsonl"
    if not p.exists():
        return
    ev = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not ev:
        return
    spaces: dict[str, list[dict]] = {}
    for e in ev:
        if e["type"] == "SPACE_PUBLISHED":
            sp = e["payload"]["space"]
            spaces.setdefault(sp["candidate_id"], []).append(sp)
    trials: dict[str, list[dict]] = {}
    for e in ev:
        if e["type"] == "TRIAL_DONE":
            t = e["payload"].get("trial") or e["payload"]
            trials.setdefault(t["space_id"], []).append(t)

    for e in ev:
        if e["type"] != "SPACE_EXPANDED":
            continue
        cid = e["payload"]["candidate_id"]
        seq = spaces.get(cid, [])
        if len(seq) < 2:
            continue
        vmax = max(s["version"] for s in seq)
        new = [s for s in seq if s["version"] == vmax][-1]
        olds = [s for s in seq if s["version"] == vmax - 1]
        if not olds:
            continue
        old = olds[-1]
        o_ch = {d["name"]: set(d["choices"]) for d in old["domains"]}
        n_ch = {d["name"]: list(d["choices"]) for d in new["domains"]}
        pre = [t for t in trials.get(old["space_id"], [])
               if t["status"] == "complete" and t.get("latency_ms")]
        post = [t for t in trials.get(new["space_id"], [])
                if t["status"] == "complete" and t.get("latency_ms")]
        if not pre or not post:
            continue
        b_pre = min(t["latency_ms"]["mean"] for t in pre)
        bt = min(post, key=lambda t: t["latency_ms"]["mean"])
        gain = 100.0 * (b_pre - bt["latency_ms"]["mean"]) / b_pre
        for k in e["payload"].get("knobs", []):
            nm, d = k["name"], k["direction"]
            added = [v for v in n_ch.get(nm, []) if v not in o_ch.get(nm, set())]
            if not added:
                continue  # requested but the agent declined to widen it
            yield (ev[0]["ts"], cid, d, nm, added,
                   bt["params"]["values"].get(nm) in added, gain)


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    fix_ts = commit_ts(HARD_EDGE_COMMIT)
    rows = [r for run in sorted(runs_dir.iterdir()) if run.is_dir()
            for r in expansions(run)]
    if not rows:
        print("no expansions with a measurable before/after found")
        return 0

    def report(label: str, subset: list) -> None:
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for _ts, _cid, d, _nm, _added, used, _gain in subset:
            agg[d][0] += 1
            agg[d][1] += int(used)
        print(f"\n{label}")
        for d in ("max", "min"):
            n, w = agg[d]
            if n:
                print(f"  direction={d:4s}  knobs widened {n:3d}  "
                      f"best used an added value: {w:3d}  ({100 * w / n:.0f}%)")

    report(f"ALL RUNS ({len(rows)} widened knobs)", rows)
    if fix_ts:
        post = [r for r in rows if r[0] >= fix_ts]
        print(f"\n(hard-edge filter {HARD_EDGE_COMMIT} landed "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(fix_ts))})")
        report(f"POST-FILTER ONLY ({len(post)} widened knobs)", post)
        misses = [r for r in post if r[2] == "min" and not r[5]]
        if misses:
            print(f"\n  the {len(misses)} post-filter downward expansions that missed:")
            for _ts, cid, _d, nm, added, _used, _g in misses:
                print(f"    {cid}  {nm:18s} added={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
