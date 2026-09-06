"""How did each completed run actually end, and how much budget did it leave?

Written after D4 (an outer loop that could not terminate) to answer the general form of the
question: is `RUN_FINISHED` reached for a *good* reason, or does the loop keep finding ways to
stop with the clock unspent? Reads only.

Termination taxonomy, from the last global CONVERGENCE_DECIDED:
  wall_clock       elapsed >= wall_clock_hours          <- the budget was actually used
  nothing_active   every family frozen                  <- may be premature; the freeze
                                                           reasons say which
  stuck            OUTER_LOOP_STUCK (the D4 guard)      <- a defect
  killed           no RUN_FINISHED at all

Usage: python scripts/audit_run_termination_reasons.py [runs_dir]
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def classify(run: pathlib.Path) -> dict | None:
    ev_path = run / "events.jsonl"
    if not ev_path.exists():
        return None
    finished = None
    last_global = None
    stuck = False
    freezes: dict[str, str] = {}
    unrewritable = 0
    novelty_calls = 0
    first_ts = last_ts = None
    with ev_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:            # noqa: BLE001 - a truncated tail must not abort
                continue
            if first_ts is None:
                first_ts = e.get("ts")
            last_ts = e.get("ts")
            t = e["type"]
            pl = e.get("payload") or {}
            if t == "RUN_FINISHED":
                finished = pl.get("summary") or {}
            elif t == "OUTER_LOOP_STUCK":
                stuck = True
            elif t == "FAMILY_FROZEN_UNREWRITABLE":
                unrewritable += 1
            elif t == "AGENT_CALL_STARTED" and pl.get("module") == "novelty":
                novelty_calls += 1
            elif t == "CONVERGENCE_DECIDED":
                d = pl.get("decision") or {}
                if d.get("scope") == "global":
                    last_global = d
                elif d.get("scope") == "family" and d.get("verdict") == "freeze":
                    fid = pl.get("family_id")
                    if fid:
                        freezes[fid] = d.get("stop_kind") or "?"

    try:
        man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        wc = ((man.get("config") or {}).get("budgets") or {}).get("wall_clock_hours") or 0
    except Exception:                    # noqa: BLE001
        wc = 0
    elapsed = ((last_ts - first_ts) / 3600) if (first_ts and last_ts) else 0.0

    if stuck:
        why = "stuck (D4 guard)"
    elif finished is None:
        why = "killed / no RUN_FINISHED"
    elif last_global and last_global.get("stop_kind") == "budget_exhausted" and wc \
            and elapsed >= wc * 0.98:
        why = "wall_clock"
    elif last_global and last_global.get("verdict") == "freeze":
        why = f"nothing_active ({last_global.get('stop_kind')})"
    else:
        why = "unknown"

    return {
        "run": run.name,
        "why": why,
        "elapsed_h": round(elapsed, 2),
        "budget_h": wc,
        "used_pct": round(elapsed / wc * 100, 1) if wc else None,
        "freeze_kinds": sorted(set(freezes.values())),
        "unrewritable_frozen": unrewritable,
        "novelty_calls": novelty_calls,
    }


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    rows = [r for r in (classify(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    if not rows:
        print(f"no runs under {root}")
        return 1
    print(f"{'run':38s} {'why':26s} {'elapsed':>8s} {'used%':>6s} {'nov':>4s} freeze kinds")
    print("-" * 108)
    for r in rows:
        print(f"{r['run']:38s} {r['why']:26s} {r['elapsed_h']:>7.2f}h "
              f"{str(r['used_pct'] or '-'):>6s} {r['novelty_calls']:>4d} "
              f"{','.join(r['freeze_kinds']) or '-'}"
              + (f"  [{r['unrewritable_frozen']} unrewritable]"
                 if r["unrewritable_frozen"] else ""))

    print()
    early = [r for r in rows if r["used_pct"] is not None and r["used_pct"] < 60
             and r["why"].startswith("nothing_active")]
    print(f"{len(early)} of {len(rows)} runs ended with <60% of the clock used and every "
          "family frozen.")
    if early:
        print("  Those are the runs where a freeze rule, not the budget, decided the "
              "ending -- worth reading one by one:")
        for r in early:
            print(f"    {r['run']}  {r['used_pct']}%  freezes={','.join(r['freeze_kinds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
