"""Does a RESOURCE_WALL_ATTRIBUTED event mean a wall was attributed? No -- check the verdict.

WHY. An inline `grep -c RESOURCE_WALL_ATTRIBUTED` over the S7 pair returns 6 and 8, while the
run monitor reports zero attributed walls. Both cannot be right, and the difference decides
whether "S7 fired 25 times and enqueued nothing because nothing was attributable" is a true
sentence or a reading error on my side.

The event name is the name of the STEP, not of a positive outcome: the probe runs and records
what it found, including `not_attributed`. So the count of events is an upper bound on the count
of attributed walls, and quoting it as the latter overstates C2's raw material.

Prints, per run: how many events, how many walls inside them, and the verdict histogram -- plus
the over_ratio for every wall that failed, because a ratio below 1 means shared-memory use at
theta* never reached the limit, i.e. the refusal had some other cause.

Also avoids `grep -c || echo 0`, which prints "0\\n0" on an empty match and has bitten this
project in both directions.
"""
from __future__ import annotations

import json
import os
import sys


def _read(path: str) -> list[dict]:
    ev = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return ev


def audit(events: list[dict], label: str, quiet: bool = False) -> dict:
    def _say(*a: object) -> None:
        if not quiet:
            print(*a)

    out = {"events": 0, "walls": 0, "verdicts": {}, "attributed": 0, "dropped_by_slope": 0,
           "kept_by_slope": 0, "attributed_and_kept": 0, "attributed_but_dropped": 0,
           "failed_ratios": []}
    for e in events:
        if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
            continue
        out["events"] += 1
        p = e.get("payload") or {}
        for w in (p.get("walls") or []):
            out["walls"] += 1
            # The slope filter runs FIRST. A wall it dropped carries no verdict because it was
            # never a candidate for probing, so counting it under "(no verdict field)" conflates
            # "the slope filter dropped it" with "it passed the filter and was never probed" --
            # two different next actions. Classify by the filter before reading the verdict.
            #
            # But a dropped wall can still carry `verdict: attributed` from an earlier probe, and
            # that case matters: attributed yet never delivered. Count it separately instead of
            # letting the `continue` hide it.
            kept = bool(w.get("monotone")) and (w.get("tail_gain_pct") or 0) > 0
            if not kept:
                out["dropped_by_slope"] += 1
                if str(w.get("verdict") or "") == "attributed":
                    out["attributed"] += 1
                    out["attributed_but_dropped"] += 1
                continue
            out["kept_by_slope"] += 1
            v = str(w.get("verdict") or "(never probed: no verdict field)")
            out["verdicts"][v] = out["verdicts"].get(v, 0) + 1
            if v == "attributed":
                out["attributed"] += 1
                out["attributed_and_kept"] += 1
            else:
                r = w.get("over_ratio")
                out["failed_ratios"].append((str(w.get("param") or "?"), v, r))

    _say("  %-42s events %d  walls %d" % (label, out["events"], out["walls"]))
    if out["walls"]:
        _say("      slope filter: kept %d  dropped %d"
             % (out["kept_by_slope"], out["dropped_by_slope"]))
        if out["verdicts"]:
            _say("      verdicts among the KEPT: %s" % ", ".join(
                "%s=%d" % kv for kv in sorted(out["verdicts"].items(), key=lambda kv: -kv[1])))
        _say("      attributed %d   past BOTH gates %d   attributed but slope-dropped %d"
             % (out["attributed"], out["attributed_and_kept"], out["attributed_but_dropped"]))
        for param, v, r in out["failed_ratios"]:
            rs = ("%.3f" % r) if isinstance(r, (int, float)) else "?"
            _say("      failed: %-14s %-30s over_ratio %s%s"
                 % (param, v, rs,
                    "  (<1 => use at theta* never reached the limit)"
                    if isinstance(r, (int, float)) and r < 1 else ""))
    return out


def _selftest() -> int:
    def _ev(walls):
        return {"type": "RESOURCE_WALL_ATTRIBUTED", "payload": {"walls": walls}}

    ok = True
    # An event whose walls all failed must NOT count as attributed. This is the whole point:
    # counting events would report 1 attributed wall where there are none.
    f = audit([_ev([{"param": "BLOCK_M", "verdict": "not_attributed", "over_ratio": 0.727,
                     "monotone": True, "tail_gain_pct": 5.0}])], "st", quiet=True)
    if f["events"] != 1 or f["attributed"] or f["walls"] != 1:
        print("FAIL: a failed wall was counted as attributed (%s)" % f)
        ok = False
    # A wall past BOTH gates must be visible as such, or the probe could not tell the real
    # zero from a broken reader.
    f = audit([_ev([{"param": "BLOCK_N", "verdict": "attributed", "over_ratio": 1.29,
                     "monotone": True, "tail_gain_pct": 12.0}])], "st", quiet=True)
    if f["attributed"] != 1 or f["attributed_and_kept"] != 1:
        print("FAIL: a wall past both gates was not seen (%s)" % f)
        ok = False
    # Attributed but dropped by the slope filter: attributed=1, both gates=0. Collapsing these
    # would let a wall that never reached a prompt read as one that did.
    f = audit([_ev([{"param": "X", "verdict": "attributed", "over_ratio": 1.1,
                     "monotone": False, "tail_gain_pct": 12.0}])], "st", quiet=True)
    if f["attributed"] != 1 or f["attributed_and_kept"] or f["attributed_but_dropped"] != 1:
        print("FAIL: slope-dropped wall conflated with a delivered one (%s)" % f)
        ok = False
    # A wall the slope filter dropped with NO verdict must be counted as slope-dropped, not as
    # "never probed" -- it was never eligible. This is the distinction the analyzer's three-way
    # split rests on, so it needs its own control here too.
    f = audit([_ev([{"param": "Z", "monotone": False, "tail_gain_pct": 0.0}])], "st", quiet=True)
    if f["dropped_by_slope"] != 1 or f["verdicts"] or f["kept_by_slope"]:
        print("FAIL: a slope-dropped wall was read as a verdict outcome (%s)" % f)
        ok = False
    # An event carrying no walls must count as an event and no walls -- that asymmetry is
    # exactly what the inline grep hid.
    f = audit([_ev([])], "st", quiet=True)
    if f["events"] != 1 or f["walls"]:
        print("FAIL: an empty event was miscounted (%s)" % f)
        ok = False
    # A wall with no verdict field must not default to attributed.
    f = audit([_ev([{"param": "Y", "monotone": True, "tail_gain_pct": 3.0}])], "st", quiet=True)
    if f["attributed"] or "(never probed: no verdict field)" not in f["verdicts"]:
        print("FAIL: a missing verdict defaulted to attributed (%s)" % f)
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    print("EVENT COUNT IS AN UPPER BOUND ON ATTRIBUTED WALLS -- the event is the step, not the")
    print("outcome. Numbers below separate the two.")
    print()
    tot = {"events": 0, "walls": 0, "attributed": 0, "attributed_and_kept": 0}
    for path in paths:
        if not os.path.exists(path):
            continue
        f = audit(_read(path), "/".join(path.split(os.sep)[-3:-1]))
        for k in tot:
            tot[k] += f[k]
    print()
    print("POOLED  events %d  walls %d  attributed %d  past BOTH gates %d"
          % (tot["events"], tot["walls"], tot["attributed"], tot["attributed_and_kept"]))
    if tot["events"] and not tot["attributed"]:
        print("  => every event recorded a probe that attributed NOTHING. Quoting the event")
        print("     count as C2's raw material overstates it by %dx." % tot["events"])
