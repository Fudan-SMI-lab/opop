"""Did the D1/D2/D3 fixes change what a Loop D run actually does?

Compares the two L1 novelty smoke runs:
  run-l1-19-20260906-183211  BEFORE the fixes (D2 live evidence: ended after 1 acceptance)
  run-l1-19-20260906-192759  AFTER  the fixes

The claim under test is narrow and falsifiable: before the fixes a novelty miss froze every
active family, so `global_verdict` ended the run one attempt after the last acceptance,
however much budget remained. After the fixes the run should keep going and spend its family
budget.

Reads only. Usage: python scripts/audit_loop_d_fix_effect.py [<before-run> <after-run>]
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BEFORE = "runs/run-l1-19-20260906-183211"
AFTER = "runs/run-l1-19-20260906-192759"


def load(run: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in (run / "events.jsonl").read_text(
        encoding="utf-8", errors="replace").splitlines() if l.strip()]


def summarise(run: pathlib.Path) -> dict:
    ev = load(run)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    b = (man.get("config") or {}).get("budgets") or {}
    fin = [e for e in ev if e["type"] == "RUN_FINISHED"]
    novelty_calls = [e for e in ev
                     if e["type"] == "AGENT_CALL_STARTED"
                     and (e.get("payload") or {}).get("module") == "novelty"]
    starts = [e for e in ev if e["type"] == "NOVELTY_ROUND_STARTED"]
    accepted = [e for e in ev if e["type"] == "NOVELTY_PRODUCED"]
    rejected = [e for e in ev if e["type"] == "NOVELTY_REJECTED"]
    fams = {e["payload"]["candidate"]["family_id"] for e in ev
            if e["type"] == "CANDIDATE_REGISTERED"}
    elapsed = (ev[-1]["ts"] - ev[0]["ts"]) / 3600
    wc = b.get("wall_clock_hours") or 0
    return {
        "run": run.name,
        "finished": bool(fin),
        "total_cap": b.get("max_families_total"),
        "active_cap": b.get("max_families_active"),
        "families": len(fams),
        "novelty_calls": len(novelty_calls),
        "round_started_keys": [(e["payload"].get("step_key"),
                               e["payload"].get("productive_families")) for e in starts],
        "accepted": len(accepted),
        "rejected": [(e["payload"] or {}).get("reason") for e in rejected],
        "elapsed_h": round(elapsed, 3),
        "budget_used_pct": round(elapsed / wc * 100, 1) if wc else None,
        "exhausted_events": sum(1 for e in ev if e["type"] == "OUTER_LOOP_EXHAUSTED"),
        "best": ((fin[-1]["payload"]["summary"].get("best") or {}).get("candidate_id")
                 if fin else None),
    }


def main() -> int:
    args = sys.argv[1:]
    before_p, after_p = (pathlib.Path(args[0]), pathlib.Path(args[1])) if len(args) >= 2 \
        else (pathlib.Path(BEFORE), pathlib.Path(AFTER))
    rows = []
    for label, p in (("BEFORE fixes", before_p), ("AFTER fixes", after_p)):
        if not (p / "events.jsonl").exists():
            print(f"{label}: {p} not found")
            continue
        s = summarise(p)
        rows.append((label, s))
        print(f"\n=== {label}: {s['run']}")
        print(f"    finished              : {s['finished']}")
        print(f"    max_families_total    : {s['total_cap']}   active: {s['active_cap']}")
        print(f"    families created      : {s['families']}")
        print(f"    novelty agent calls   : {s['novelty_calls']}")
        print(f"    novelty accepted      : {s['accepted']}   rejected: {s['rejected']}")
        print(f"    NOVELTY_ROUND_STARTED : {s['round_started_keys'] or '(event did not exist)'}")
        print(f"    OUTER_LOOP_EXHAUSTED  : {s['exhausted_events']}")
        print(f"    elapsed               : {s['elapsed_h']} h "
              f"({s['budget_used_pct']}% of budget)")
        print(f"    reported best         : {s['best']}")

    if len(rows) == 2:
        (_, a), (_, b) = rows
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        # The decisive number: how many novelty attempts were REACHED, relative to how many
        # the family budget allows. Before the fixes this was capped at 1 by construction.
        for label, s in rows:
            allowed = (s["total_cap"] or 0) - 1  # seeds=1 in this smoke config
            print(f"  {label:13s} novelty calls {s['novelty_calls']} of "
                  f"{allowed} the family budget permits; "
                  f"{s['families']} families; {s['budget_used_pct']}% budget")
        if b["novelty_calls"] > a["novelty_calls"]:
            print("\n  D2/D3 CONFIRMED: the fixed run reached novelty attempts the old one "
                  "could not.")
        elif b["novelty_calls"] == a["novelty_calls"] == 1:
            print("\n  INCONCLUSIVE on this pair: both made exactly 1 novelty call. Check "
                  "whether the AFTER run's family budget actually permitted a second.")
        else:
            print("\n  NOT confirmed by this pair -- read the traces above before claiming "
                  "the fix works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
