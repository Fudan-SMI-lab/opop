"""Is either arm STALLED, and does the trial-count gap survive the free rejections?

WHY NOW, mid-run. Two things showed up in the event histogram that cannot wait for the end of the run:

  1. The treatment arm has 8 AGENT_CALL_STARTED and 7 AGENT_CALL_FINISHED. One call is either in flight
     or died without an event. An agent call has an 1800 s whole-call deadline, and while it runs the arm
     tunes NOTHING -- so a hung call spends the treatment arm's remaining budget on nothing while the
     control arm keeps tuning. That widens the very gap we are trying to attribute, and it is a defect,
     not a property of the independent variable.
  2. The control arm logged 42 CONFIG_SCREENED_INFEASIBLE against the treatment arm's 14. Prescreened
     configs are SAVINGS, not failures (`prescreened-configs-are-savings-not-failures`), and they cost
     almost nothing -- so if they are counted inside the trial totals, part of the 1.58x "trial rate gap"
     is free rejections and not search at all. That changes what the gap means.

WHY THIS DOES NOT GUESS FIELD NAMES. A first attempt to read timestamps with `"ts"` printed nothing --
the constant/empty-reading-is-a-broken-probe shape. So this dumps the actual top-level keys of a real
event and picks the timestamp field from what is THERE, and prints which field it used, so a wrong pick
is visible in the output instead of silently producing plausible elapsed times.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/inflight_and_stage.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"
ARMS = ("s7-treatment", "s7-control")

# Candidates for the timestamp field, in preference order. Whichever is present in a real event wins,
# and the choice is printed.
_TS_KEYS = ("timestamp", "ts", "time", "created_at", "wall_ts", "at")


def _load(arm: str) -> list[dict]:
    out = []
    path = BASE / arm / RUN / "events.jsonl"
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _parse(v) -> float | None:
    """Seconds since epoch from whatever shape the emitter used, or None."""
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _ts_key(events: list[dict]) -> str | None:
    for e in events:
        for k in _TS_KEYS:
            if k in e and _parse(e[k]) is not None:
                return k
    return None


def main() -> int:
    now = datetime.now(timezone.utc).timestamp()

    for arm in ARMS:
        ev = _load(arm)
        key = _ts_key(ev)
        print("=" * 78)
        print("%s   %d events" % (arm, len(ev)))
        if key is None:
            print("  NO USABLE TIMESTAMP FIELD. Top-level keys of the first event:")
            print("   ", sorted(ev[0].keys()) if ev else "(no events)")
            print("  Every duration below would be invented, so none are printed.")
        else:
            print("  timestamp field = %r  (keys: %s)" % (key, ", ".join(sorted(ev[0].keys()))))
            ts = [t for t in (_parse(e.get(key)) for e in ev) if t is not None]
            if ts:
                span = max(ts) - min(ts)
                print("  event span %.2f h   last event %.1f s ago" % (span / 3600.0, now - max(ts)))

        # ---- in-flight agent calls: STARTED with no FINISHED/FAILED for the same call id.
        # The id field is unknown, so match on whatever id-like key both events share.
        started: dict[str, dict] = {}
        closed: set[str] = set()
        id_keys = ("call_id", "agent_call_id", "id")
        for e in ev:
            t = e.get("type") or ""
            if not t.startswith("AGENT_CALL_"):
                continue
            p = e.get("payload") or {}
            cid = None
            for k in id_keys:
                if p.get(k):
                    cid = str(p[k])
                    break
            if cid is None:
                cid = "?%d" % len(started)
            if t.endswith("STARTED"):
                started[cid] = e
            else:
                closed.add(cid)

        open_calls = [(cid, e) for cid, e in started.items() if cid not in closed]
        print("  agent calls: %d started, %d closed, %d OPEN" % (
            len(started), len(closed), len(open_calls)))
        for cid, e in open_calls:
            p = e.get("payload") or {}
            age = ""
            if key is not None:
                t = _parse(e.get(key))
                if t is not None:
                    age = "  age %.0f s" % (now - t)
            print("    OPEN  %-26s module=%s%s" % (
                cid[:26], p.get("module") or p.get("agent") or p.get("name") or "?", age))
            if age and (now - (_parse(e.get(key)) or now)) > 1800:
                print("      PAST THE 1800 s WHOLE-CALL DEADLINE -- this is a stall, not a slow call.")
            print("      payload keys: %s" % ", ".join(sorted(p.keys())))

        # ---- does the trial total include the free rejections?
        trial_ids: set[str] = set()
        for e in ev:
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or (e.get("payload") or {})
            tid = tr.get("trial_id") or tr.get("id")
            if tid:
                trial_ids.add(str(tid))
        screened_ids: set[str] = set()
        # Print the key list ONCE. A per-event print here floods the screen and scrolls the answer off
        # it -- the recorded `a-probe-that-names-its-target-matches-itself` failure was the same shape:
        # the probe's own output destroyed the evidence it was collecting.
        shown_screen_keys = False
        n_screened = 0
        for e in ev:
            if e.get("type") != "CONFIG_SCREENED_INFEASIBLE":
                continue
            n_screened += 1
            p = e.get("payload") or {}
            tid = p.get("trial_id") or p.get("id")
            if tid:
                screened_ids.add(str(tid))
            elif not shown_screen_keys:
                shown_screen_keys = True
                print("  CONFIG_SCREENED_INFEASIBLE carries no trial id; payload keys: %s"
                      % ", ".join(sorted(p.keys())))
                print("    (%d such events; no id means they cannot be inside the trial total)"
                      % n_screened)
        overlap = trial_ids & screened_ids
        print("  TRIAL_DONE=%d  screened=%d  ids shared with trials=%d  =>  %s" % (
            len(trial_ids), n_screened, len(overlap),
            "screened configs ARE inside the trial total (so the gap is partly free rejections)"
            if overlap else "screened configs are SEPARATE from the trial total"))

        # ---- loop stage: has anything downstream of tuning happened yet?
        for t in ("SPACE_EXPANDED", "REWRITE_PRODUCED", "CONVERGENCE_DECIDED",
                  "REPAIR_PRODUCED", "NOVELTY_PRODUCED", "RUN_FINISHED"):
            n = sum(1 for e in ev if e.get("type") == t)
            if t == "SPACE_EXPANDED" and n:
                for e in ev:
                    if e.get("type") != t:
                        continue
                    p = e.get("payload") or {}
                    print("  %-20s %s" % (t, {k: p[k] for k in sorted(p)
                                             if k in ("param", "space_id", "added", "new_choices",
                                                      "reason", "candidate_id")}))
            else:
                print("  %-20s %d" % (t, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
