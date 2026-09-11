"""How much wall clock does each run actually have, and does a resume change it?

`_elapsed_hours()` measures from `self.t0 = time.monotonic()`, set in `Orchestrator.__init__`.
`cmd_resume` goes through `build_orchestrator`, which constructs a new Orchestrator -- so a RESUMED
run's clock restarts at zero and it gets a fresh `wall_clock_hours` on top of whatever it already
spent. That is not obviously wrong (the point of a resume is to continue the work, and the trials
already on disk are not re-run), but it means "the three runs had 12 h each" is false for a resumed
one, and any statement comparing what the arms achieved "in the same wall clock" has to say so.

This reads the two clocks separately from events.jsonl:

  * WALL age  -- first event to last event, which is what a reader assumes "elapsed" means and what
    the wall-clock budget is NOT measured against on a resumed run.
  * BUDGET age -- from the last RUN_INTERRUPTED (or the first event if there is none) to the last
    event, which is what `_elapsed_hours()` actually sees.

Prints both plus the remaining budget, and flags the divergence rather than silently choosing one.
The distinction is only visible from disk: no event carries `elapsed_hours` except the ones written
when the budget fires, which is exactly when it is too late to notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read(rd: Path) -> dict:
    evs = []
    with (rd / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e.get("ts"), (int, float)):
                evs.append(e)
    if not evs:
        raise SystemExit("no timestamped events in %s" % rd)

    first, last = evs[0]["ts"], evs[-1]["ts"]
    # The clock the budget actually uses restarts at the process that is running now. The last
    # RUN_INTERRUPTED marks the end of the previous process; everything after it belongs to the
    # current one. A run that was never interrupted has one process and the two clocks agree.
    resume_at = None
    interrupts = 0
    for e in evs:
        if e.get("type") == "RUN_INTERRUPTED":
            interrupts += 1
            resume_at = e["ts"]
    # The first event AFTER the interrupt is when the new process began appending.
    budget_t0 = first
    if resume_at is not None:
        later = [e["ts"] for e in evs if e["ts"] > resume_at]
        budget_t0 = min(later) if later else resume_at

    # Reported elapsed_hours, when any event carries it -- the only in-band confirmation available.
    reported = [(e.get("type"), (e.get("payload") or {}).get("elapsed_hours"))
                for e in evs if isinstance((e.get("payload") or {}).get("elapsed_hours"),
                                           (int, float))]
    return {"wall_h": (last - first) / 3600.0,
            "budget_h": (last - budget_t0) / 3600.0,
            "interrupts": interrupts,
            "reported_elapsed": reported[-3:]}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    budget = None
    rows = []
    for lbl, rd in zip(argv[0::2], argv[1::2]):
        p = Path(rd)
        # The configured budget lives in manifest.json, not in RUN_CREATED.
        man = p / "manifest.json"
        b = None
        if man.exists():
            try:
                m = json.loads(man.read_text(encoding="utf-8"))
                b = (((m.get("config") or m).get("budgets") or {}).get("wall_clock_hours"))
                if b is None:
                    # The manifest nests differently across versions; find the key anywhere.
                    txt = man.read_text(encoding="utf-8")
                    import re
                    hit = re.search(r'"wall_clock_hours":\s*([0-9.]+)', txt)
                    b = float(hit.group(1)) if hit else None
            except ValueError:
                b = None
        r = read(p)
        r["configured_budget_h"] = b
        budget = b if budget is None else budget
        rows.append((lbl, r))

    print("%-10s %10s %10s %10s %7s  %s" % (
        "arm", "wall_h", "budget_h", "left_h", "intr", "note"))
    for lbl, r in rows:
        b = r["configured_budget_h"]
        left = (b - r["budget_h"]) if b else float("nan")
        note = ""
        if r["interrupts"]:
            gained = r["wall_h"] - r["budget_h"]
            note = ("RESUMED %dx: the budget clock restarted, so it has %.2f h MORE total wall "
                    "clock than a run that was never interrupted" % (r["interrupts"], gained))
        print("%-10s %10.2f %10.2f %10.2f %7d  %s" % (
            lbl, r["wall_h"], r["budget_h"], left, r["interrupts"], note))
        if r["reported_elapsed"]:
            print("%-10s   in-band elapsed_hours seen: %s" % ("", r["reported_elapsed"]))

    resumed = [l for l, r in rows if r["interrupts"]]
    if resumed and len(rows) > 1:
        print("\n!! Report the budget asymmetry beside any cross-arm claim: %s was resumed, so "
              "'the same wall clock' is not true of it. The trials already on disk are not re-run, "
              "so the extra time buys extra SEARCH, which is the confound "
              "`check_arm_search_parity.py` measures." % ", ".join(resumed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
