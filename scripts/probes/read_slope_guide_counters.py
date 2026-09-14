"""Slope-guide counters, read the way the emitter intends: LAST snapshot per space, never summed.

WHY THIS EXISTS. `SlopeGuide.snapshot()` says it in its own docstring -- "THE COUNTERS ARE NOT A
PARTITION and must not be summed" -- and I summed them anyway, reporting `n_skipped_no_wall = 158`,
`n_suggested = 15` and `no_value_toward_wall = 24` for a run whose real values were 63, 7 and 13. Two
independent reasons, both of which the summing hits:

  1. THE COUNTERS ARE CUMULATIVE PER INSTANCE, and there is one `SlopeGuide` per tuning pass. Within one
     space the journalled value climbs 1, 2, 3, 4 as `n_recomputes` climbs 1, 2, 3, 4. Summing the
     events therefore sums running totals and yields a triangular number -- not a count of anything.
     Measured: 9 recomputes in one space read as 1+2+3+4 = 10 if summed over 4 events.
  2. THEY COUNT DIFFERENT UNITS AND OVERLAP. `no_wall` and `incomplete_incumbent` count RECOMPUTES;
     `no_value_toward_wall` and `already_proposed` count KNOBS, of which one recompute can decline
     several -- and one recompute can increment BOTH kinds, because a row dropped while resolving
     targets leaves an empty list that is then reported as "no wall". In-repo measurement: 751 + 194
     against 790 recomputes.

The correct read is: per space, take the LAST snapshot; then sum ACROSS spaces, which are different
instances. The self-check is that `sum(n_recomputes)` must equal the number of `SLOPE_GUIDE_STEP`
events -- a probe whose arithmetic cannot be checked is how the wrong numbers survived nine hours.

    python scripts/probes/read_slope_guide_counters.py <run_dir> [<run_dir> ...]
    python scripts/probes/read_slope_guide_counters.py --selftest

WHAT THIS DOES NOT DO: it says nothing about whether an enqueued point was any GOOD. That is
`did_the_enqueued_point_win.py`, and it is the reading that tests C2's premise rather than its plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# `n_recomputes` first: it is the denominator every other counter is read against, and it is also the
# self-check. The rest are grouped by UNIT, because mixing the two units is half the bug above.
RECOMPUTE_COUNTERS = ("n_skipped_no_wall", "n_skipped_incomplete_incumbent")
KNOB_COUNTERS = ("n_skipped_no_value_toward_wall", "n_skipped_already_proposed")
OTHER = ("n_suggested", "n_proposed_never_drawn_value")
ALL_KEYS = ("n_recomputes",) + OTHER + RECOMPUTE_COUNTERS + KNOB_COUNTERS


def steps_of(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    with (run_dir / "events.jsonl").open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue        # an in-flight run's last line can be a partial write
            if e.get("type") == "SLOPE_GUIDE_STEP":
                out.append(e.get("payload") or {})
    return out


def fold(payloads: list[dict]) -> dict:
    """Per-space last snapshot, summed across spaces, plus the self-check."""
    last: dict[str, dict] = {}
    enqueued = 0
    refused = 0
    for p in payloads:
        enqueued += len(p.get("enqueued") or [])
        refused += len(p.get("refused") or [])
        sid = str(p.get("space_id") or "?")
        prev = last.get(sid)
        # max() rather than "the last one seen": events are in order, but a space id could recur after
        # a K expansion, and taking the max of n_recomputes is right either way.
        if prev is None or (p.get("n_recomputes") or 0) >= (prev.get("n_recomputes") or 0):
            last[sid] = p
    totals = {k: sum(int(s.get(k) or 0) for s in last.values()) for k in ALL_KEYS}
    return {
        "n_steps": len(payloads),
        "n_spaces": len(last),
        "enqueued": enqueued,
        "refused": refused,
        "totals": totals,
        "consistent": totals["n_recomputes"] == len(payloads),
    }


def report(run_dir: Path) -> dict:
    res = fold(steps_of(run_dir))
    t = res["totals"]
    print("=" * 92)
    print("run: %s" % run_dir)
    print("SLOPE_GUIDE_STEP events %d   spaces with a guide %d   points ENQUEUED %d   refused %d"
          % (res["n_steps"], res["n_spaces"], res["enqueued"], res["refused"]))
    if res["n_steps"] == 0:
        print("zero steps => the slope guide is OFF in this arm. A structural zero, not a failure;")
        print("in a paired experiment this is the positive control.")
        return res
    print()
    print("  recomputes                         %d" % t["n_recomputes"])
    print("  suggestions made                   %d" % t["n_suggested"])
    print("  ...of which enqueued               %d" % res["enqueued"])
    if t["n_suggested"] and res["enqueued"] == t["n_suggested"]:
        print("      every suggestion was accepted by the sampler: ZERO refusals, so the bottleneck is")
        print("      entirely in MAKING a suggestion, not in getting one drawn.")
    print("  declined, counted in RECOMPUTES:")
    for k in RECOMPUTE_COUNTERS:
        share = (100.0 * t[k] / t["n_recomputes"]) if t["n_recomputes"] else 0.0
        print("      %-34s %4d   (%.0f%% of recomputes)" % (k, t[k], share))
    print("  declined, counted in KNOBS (one recompute can decline several; overlaps the above):")
    for k in KNOB_COUNTERS:
        print("      %-34s %4d" % (k, t[k]))
    print()
    if res["consistent"]:
        print("  self-check OK: sum(n_recomputes)=%d equals the %d SLOPE_GUIDE_STEP events."
              % (t["n_recomputes"], res["n_steps"]))
    else:
        print("  !! SELF-CHECK FAILED: sum(n_recomputes)=%d but there are %d SLOPE_GUIDE_STEP events."
              % (t["n_recomputes"], res["n_steps"]))
        print("     A space id recurred with a RESET counter, or a snapshot is missing. Do not quote")
        print("     any number above until this is explained -- the per-space fold assumes one")
        print("     monotonically climbing counter per space.")
    print("  reminder: these counters are cumulative and their units differ. Never sum them across")
    print("  events, and never add the two groups together.")
    return res


def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print("  %-70s %s" % (name, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(name)

    def step(space: str, n: int, no_wall: int, sugg: int = 0, enq: int = 0) -> dict:
        return {"space_id": space, "n_recomputes": n, "n_skipped_no_wall": no_wall,
                "n_suggested": sugg, "n_skipped_no_value_toward_wall": 0,
                "n_skipped_already_proposed": 0, "n_skipped_incomplete_incumbent": 0,
                "n_proposed_never_drawn_value": 0,
                "enqueued": [{"knob": "W", "knob_value": 8}] * enq, "refused": []}

    # THE BUG. One space, four cumulative snapshots. Summing gives 1+2+3+4=10; the truth is 4.
    r = fold([step("sp1", 1, 1), step("sp1", 2, 2), step("sp1", 3, 3), step("sp1", 4, 4)])
    check("four cumulative snapshots in one space read as 4, not 1+2+3+4=10",
          r["totals"]["n_skipped_no_wall"] == 4)
    check("recomputes likewise read as 4", r["totals"]["n_recomputes"] == 4)
    check("self-check catches nothing wrong here", r["consistent"] is True)

    # Across spaces the totals DO add, because each space is its own SlopeGuide instance.
    r = fold([step("sp1", 1, 1), step("sp1", 2, 2), step("sp2", 1, 1)])
    check("two spaces' last snapshots are summed (2 + 1 = 3)", r["totals"]["n_skipped_no_wall"] == 3)
    check("and so are their recomputes", r["totals"]["n_recomputes"] == 3)

    # The self-check must FAIL loudly when the fold's assumption breaks, or it is decoration.
    r = fold([step("sp1", 1, 1), step("sp1", 1, 1)])
    check("a space whose counter does not climb trips the self-check", r["consistent"] is False)

    # Enqueued points are counted per event, NOT folded -- they are a list of things that happened.
    r = fold([step("sp1", 1, 0, sugg=1, enq=1), step("sp1", 2, 0, sugg=2, enq=1)])
    check("enqueued points are summed over events (1 + 1 = 2)", r["enqueued"] == 2)
    check("suggestions are folded, not summed (last snapshot = 2)", r["totals"]["n_suggested"] == 2)

    # A control arm: no steps at all must read as a structural zero, not an error.
    r = fold([])
    check("no steps => zeros and no inconsistency", r["n_steps"] == 0 and r["totals"]["n_recomputes"] == 0)

    # The real S7 numbers, as a regression fixture: 72 events, 18 spaces, 7 enqueued. Folding a
    # synthetic version of that shape must not reproduce the inflated 158.
    ev = []
    for sp in range(18):
        for k in range(1, 5):
            ev.append(step("sp%d" % sp, k, k))
    r = fold(ev)
    check("18 spaces x 4 climbing snapshots = 72 recomputes, not 720",
          r["totals"]["n_recomputes"] == 72 and r["n_steps"] == 72)
    check("...and no_wall likewise 72, nowhere near the summed 180",
          r["totals"]["n_skipped_no_wall"] == 72)

    print()
    print("%d/%d checks passed" % (11 - len(fails), 11))
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.runs:
        ap.error("give at least one run dir, or --selftest")
    bad = 0
    for run in args.runs:
        res = report(run)
        if not res["consistent"] and res["n_steps"]:
            bad += 1
        print()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
