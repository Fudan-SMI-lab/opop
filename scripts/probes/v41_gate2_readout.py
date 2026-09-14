"""Gate-2 readout: the pilot's instrument-health table, from events.jsonl ONLY.

Usage:  python scripts/probes/v41_gate2_readout.py <runs_dir>/<run-id>

Reads the on-disk event log through the store reader (never re-derives), prints:
  - three rates: C4 fresh completion / sign / print-gate pass
  - the E funnel: admissions, fresh completions, distinct new values, rank-changed
  - same-stratum competition: e_decisions with eligible>=2, RR-vs-gain differed count
  - probe layer: batches, answers, conditioned walls by kind, D budget actuals
  - decision: M2a (RR pair) if competition is non-trivial, else M2b (replication pair)

Counters are read from FINAL values only (TUNING_DONE.conditional_scan snapshots), never
summed across events (the triangular-number lesson). Run AFTER RUN_FINISHED.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# The authoritative reader (G21): an empty read RAISES rather than printing a clean,
# plausible, wrong table.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kernel_optimizer.store.read import read_events  # noqa: E402


def main() -> int:
    run_dir = Path(sys.argv[1])
    finished = False
    tuning_final: list[dict] = []       # final conditional_scan snapshot per space
    blocks_done: list[dict] = []
    admitted: list[dict] = []
    probe_batches: list[dict] = []
    briefs = 0
    fresh_forced = 0

    for ev in read_events(run_dir):
        t = ev.get("type")
        p = ev.get("payload", {})
        if t == "RUN_FINISHED":
            finished = True
        elif t == "TUNING_DONE" and p.get("conditional_scan"):
            tuning_final.append(p["conditional_scan"])
        elif t == "SCAN_BLOCK_DONE":
            blocks_done.append(p)
        elif t == "SCAN_BLOCK_ADMITTED":
            admitted.append(p)
        elif t == "UW_PROBE_BATCH":
            probe_batches.append(p)
        elif t == "CONDITIONED_BRIEF_DELIVERED":
            briefs += 1
        elif t == "FRESH_MEASUREMENT_FORCED":
            fresh_forced += 1

    if not finished:
        print("!! RUN NOT FINISHED — verdicts are unreadable mid-run; aborting")
        return 2

    c4 = [b for b in blocks_done if b.get("kind") == "C4"]
    c4_full = [b for b in c4 if b.get("full")]
    c4_pos_out = [b for b in c4_full if b.get("direction") == "outward"]
    c4_pos_in = [b for b in c4_full if b.get("direction") == "inward"]
    c4_unresolved = [b for b in c4_full if b.get("direction") is None]

    print("== three rates (per attempted C4) ==")
    n_c4 = len(c4)
    print(f"C4 blocks completed:      {n_c4}")
    if n_c4:
        print(f"  fresh-complete (full):  {len(c4_full)}/{n_c4}")
    if c4_full:
        print(f"  minted outward:         {len(c4_pos_out)}/{len(c4_full)}")
        print(f"  minted inward:          {len(c4_pos_in)}/{len(c4_full)}")
        print(f"  unresolved (no mint):   {len(c4_unresolved)}/{len(c4_full)}")

    print("\n== E funnel (final snapshots only) ==")
    e_admits = [a for a in admitted if a.get("kind") in ("E1", "E2", "E4")]
    print(f"E admissions:             {len(e_admits)}")
    print(f"fresh-forced bypasses:    {fresh_forced}")
    # `enqueued` holds ScanPoint.payload() — scan_id/role/axis/order/token_id/direction.
    # It carries NO parameter values, so this counts SLOTS per axis, not distinct values;
    # the earlier "distinct new-value enqueues" label claimed a reading the payload cannot
    # support. Distinct values, if wanted, must come from the trials, not from here.
    slots = Counter()
    refused_slots = Counter()
    for a in e_admits:
        slots[a.get("axis")] += len(a.get("enqueued", []))
        refused_slots[a.get("axis")] += len(a.get("refused", []))
    print(f"E slots enqueued by axis: {dict(slots) or '(none)'}")
    print(f"E slots refused by axis:  {dict(refused_slots) or '(none)'}")
    directions = Counter(pt.get("direction") for a in e_admits
                         for pt in a.get("enqueued", []))
    print(f"E directions:             {dict(directions) or '(none)'}")
    tot_decisions = sum(s.get("scanner", {}).get("e_decisions", 0) for s in tuning_final)
    tot_comp = sum(s.get("scanner", {}).get("e_decisions_with_competition", 0)
                   for s in tuning_final)
    tot_diff = sum(s.get("scanner", {}).get("rr_gain_differed", 0) for s in tuning_final)
    # These three are summed across SPACES (one final snapshot each), which is the one
    # legitimate summation: each TUNING_DONE carries that space's terminal value, so the
    # sum is a run total. Summing WITHIN a space across events is what produces triangular
    # numbers, and is what the per-event reads above deliberately avoid.
    print(f"E decisions (sum over {len(tuning_final)} space snapshots): {tot_decisions}")
    print(f"  with competition >=2:   {tot_comp}")
    print(f"  RR vs gain differed:    {tot_diff}")

    print("\n== probe layer ==")
    print(f"batches dispatched:       {len(probe_batches)}")
    if probe_batches:
        # EVERY batch, not the last one. Each UW_PROBE_BATCH is scoped to ONE
        # candidate/space, so `probe_batches[-1]` reports the last space's walls and calls
        # them the run's -- a reading that gets quieter the more spaces the run publishes.
        # Walls are deduplicated on their own identity (a re-planned batch for the same
        # space can re-report a wall it already found), so this is a SET of walls, never a
        # sum over events: the counters-are-cumulative lesson applies to counts, and
        # summing wall lists across batches inflates the same way.
        seen_walls: dict[tuple, dict] = {}
        for b in probe_batches:
            for w in b.get("walls", []):
                key = (b.get("space_id"), w.get("kind"), w.get("axis"),
                       w.get("partner_key"), str(w.get("refused_value")))
                seen_walls.setdefault(key, w)
        kinds = Counter(w.get("kind") for w in seen_walls.values())
        spaces_with_walls = len({k[0] for k in seen_walls})
        spaces_probed = len({b.get("space_id") for b in probe_batches})
        print(f"distinct conditioned walls:     {dict(kinds) or '(none)'}")
        print(f"  in {spaces_with_walls} of {spaces_probed} probed space(s)")
        # Soft walls cannot come from this layer: compile-only probes run ptxas-free, so
        # n_regs/n_spills are None at warmup. hard-only is the EXPECTED shape here, not a
        # gap in the mechanism.
        if kinds and not kinds.get("soft"):
            print("  (soft=0 is expected: compile-only probes cannot see n_spills)")
        witnesses = sum(len(b.get("corner_witnesses", [])) for b in probe_batches)
        print(f"corner witnesses:         {witnesses}")
        # D budget is per-space too. Report the spread, not one space's numbers.
        fr = [b["budget"]["frozen_d"] for b in probe_batches
              if b.get("budget", {}).get("frozen_d") is not None]
        retried = sum(1 for b in probe_batches if b.get("budget", {}).get("retried"))
        stopped = sum(1 for b in probe_batches if b.get("budget", {}).get("stopped"))
        if fr:
            print(f"D budget frozen_d:        min {min(fr):.1f}  max {max(fr):.1f}  "
                  f"n={len(fr)}")
        print(f"  batches retried:        {retried}")
        print(f"  batches stopped:        {stopped}")
        ans = sum(b.get("n_answered", 0) for b in probe_batches)
        disp = sum(b.get("n_dispatched", 0) for b in probe_batches)
        if disp:
            print(f"probe answer rate:        {ans}/{disp} ({100.0 * ans / disp:.1f}%)")
    print(f"conditioned briefs delivered: {briefs}   (history: 0/0/0 — any >0 is new)")

    print("\n== gate-2 decision input ==")
    if tot_comp == 0:
        print("competition frequency = 0 -> M2b (replication pair);"
              " ranking value unidentifiable in this sample (NOT evidence against C2)")
    else:
        print(f"competition frequency = {tot_comp}/{tot_decisions} -> M2a viable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
