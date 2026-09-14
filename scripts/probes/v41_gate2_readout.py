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
    new_values = Counter()
    for a in e_admits:
        for pt in a.get("enqueued", []):
            new_values[a.get("axis")] += 1
    print(f"distinct new-value enqueues by axis: {dict(new_values) or '(none)'}")
    tot_decisions = sum(s.get("scanner", {}).get("e_decisions", 0) for s in tuning_final)
    tot_comp = sum(s.get("scanner", {}).get("e_decisions_with_competition", 0)
                   for s in tuning_final)
    tot_diff = sum(s.get("scanner", {}).get("rr_gain_differed", 0) for s in tuning_final)
    print(f"E decisions:              {tot_decisions}")
    print(f"  with competition >=2:   {tot_comp}")
    print(f"  RR vs gain differed:    {tot_diff}")

    print("\n== probe layer ==")
    print(f"batches dispatched:       {len(probe_batches)}")
    if probe_batches:
        last_walls = probe_batches[-1].get("walls", [])
        kinds = Counter(w.get("kind") for w in last_walls)
        print(f"conditioned walls (last batch): {dict(kinds) or '(none)'}")
        witnesses = sum(len(b.get("corner_witnesses", [])) for b in probe_batches)
        print(f"corner witnesses:         {witnesses}")
        budgets = [b.get("budget", {}) for b in probe_batches if b.get("budget")]
        if budgets:
            print(f"D budget (last): {budgets[-1]}")
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
