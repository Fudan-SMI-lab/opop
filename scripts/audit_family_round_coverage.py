"""Did every family actually receive a rewrite round?

`active_families()` caps how many families get a rewrite round at
`max_families_active`. Before 81cd562 it ranked by incumbent latency, which is
sticky: the same top-K were re-selected every round until they exhausted their
budget, and the rest were never handed to the rewriter at all -- while still
being reported as `frozen_budget`, i.e. indistinguishable from "explored".

This script counts rewrite rounds per family per run, so the coverage effect of
that fix (or any future change to the ranking) is visible on the event log rather
than argued from the code. It also dates each run against the fix commit, because
pooling pre- and post-fix runs produces a scary number that means nothing.

A family showing 0 rounds is not automatically starved -- the loop correctly
declines to rewrite a family whose candidates never produced a correct result
(`best is None`), so that case is reported separately.

Usage: python scripts/audit_family_round_coverage.py [runs_dir]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# The commit that replaced latency ranking with unproven-first + improvement slope.
FIX_COMMIT = "81cd562"


def fix_timestamp() -> float | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", FIX_COMMIT],
            capture_output=True, text=True, timeout=15,
            cwd=str(Path(__file__).resolve().parents[1]))
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def load(run: Path) -> list[dict]:
    p = run / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    fix_ts = fix_timestamp()

    rows = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        ev = load(run)
        if not ev:
            continue
        fams: set[str] = set()
        for e in ev:
            if e["type"] == "CANDIDATE_REGISTERED":
                c = e["payload"].get("candidate", e["payload"])
                if c.get("family_id"):
                    fams.add(c["family_id"])
        if not fams:
            continue
        rounds: dict[str, int] = {}
        for e in ev:
            if e["type"] == "FAMILY_ROUND_RECORDED":
                rounds[e["payload"]["family_id"]] = \
                    rounds.get(e["payload"]["family_id"], 0) + 1
        # families that were never SELECTED at all (no family-scope decision)
        considered: set[str] = set()
        for e in ev:
            if e["type"] == "CONVERGENCE_DECIDED":
                d = e["payload"].get("decision", {})
                if d.get("scope") == "family" and e["payload"].get("family_id"):
                    considered.add(e["payload"]["family_id"])
        # families with no correct candidate: the loop declines these on purpose
        has_best: set[str] = set()
        cand_fam = {}
        for e in ev:
            if e["type"] == "CANDIDATE_REGISTERED":
                c = e["payload"].get("candidate", e["payload"])
                cand_fam[c.get("candidate_id")] = c.get("family_id")
        for e in ev:
            if e["type"] == "TRIAL_DONE":
                t = e["payload"].get("trial") or e["payload"]
                if t.get("status") == "complete" and t.get("latency_ms"):
                    f = cand_fam.get(t.get("candidate_id"))
                    if f:
                        has_best.add(f)

        never = sorted(f for f in fams if f not in considered)
        no_result = sorted(f for f in fams if f not in has_best)
        finished = any(e["type"] == "RUN_FINISHED" for e in ev)
        rows.append({
            "run": run.name.replace("run-", "").replace("20260", "0"),
            "started": ev[0]["ts"],
            "hours": (ev[-1]["ts"] - ev[0]["ts"]) / 3600.0,
            "finished": finished,
            "n_fams": len(fams),
            "counts": sorted((rounds.get(f, 0) for f in fams), reverse=True),
            "never_selected": never,
            "no_correct_result": no_result,
            "post_fix": (fix_ts is not None and ev[0]["ts"] >= fix_ts),
        })

    if not rows:
        print("no runs with families found")
        return 0

    if fix_ts:
        print(f"ranking fix {FIX_COMMIT} landed "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(fix_ts))}\n")
    else:
        print(f"(could not resolve {FIX_COMMIT}; not splitting pre/post)\n")

    hdr = (f"{'run':24s} {'hours':>6s} {'fin':>4s} {'fams':>5s} "
           f"{'rounds per family':22s} {'never sel':>9s} {'no result':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["started"]):
        mark = "" if r["post_fix"] else "  <- pre-fix"
        print(f"{r['run']:24s} {r['hours']:>6.2f} {str(r['finished'])[:1]:>4s} "
              f"{r['n_fams']:>5d} {str(r['counts']):22s} "
              f"{len(r['never_selected']):>9d} {len(r['no_correct_result']):>9d}{mark}")

    for label, want in (("PRE-FIX", False), ("POST-FIX", True)):
        g = [r for r in rows if r["post_fix"] is want]
        if not g:
            continue
        fam_n = sum(r["n_fams"] for r in g)
        never = sum(len(r["never_selected"]) for r in g)
        starved = sum(len([f for f in r["never_selected"]
                           if f not in r["no_correct_result"]]) for r in g)
        print(f"\n{label}: {len(g)} runs, {fam_n} families")
        print(f"  never selected for a rewrite round: {never} "
              f"({100 * never / fam_n:.0f}%)")
        print(f"    of which had a correct result (genuinely starved): {starved}")
        print(f"    of which never produced one (correctly declined):  {never - starved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
