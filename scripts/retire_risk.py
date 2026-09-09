"""Price the over-blocking risk of "retire a categorical value after N consecutive failures".

For every (candidate, knob, value) in a run, walk trials in event order and record how many
failures came BEFORE the first success. If a value had >= N leading failures and then
succeeded, retirement at threshold N would have killed a value that works.

This is the negative control for my own proposal, run against real data instead of a
synthetic scenario.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = [Path(a) for a in sys.argv[1:]]
THRESHOLDS = (6, 8, 12, 20, 40)

grand = {n: [0, 0] for n in THRESHOLDS}
grand_best_killed = {n: [] for n in THRESHOLDS}

for run in runs:
    ev = run / "events.jsonl"
    if not ev.exists():
        continue
    seq = defaultdict(list)
    best_overall = None
    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            ok = (tr.get("status") or "").lower() == "complete"
            cand = tr.get("candidate_id")
            lat = (tr.get("latency_ms") or {}).get("median")
            if ok and isinstance(lat, (int, float)):
                if best_overall is None or lat < best_overall:
                    best_overall = lat
            for k, v in ((tr.get("params") or {}).get("values") or {}).items():
                seq[(cand, k, str(v))].append((ok, lat if ok else None))

    print("=== %s   (best tuned overall: %s)" % (run.name, best_overall))
    per_n = {n: [0, 0] for n in THRESHOLDS}
    killed_detail = defaultdict(list)
    for (cand, knob, val), trials in seq.items():
        lead = 0
        for ok, _ in trials:
            if ok:
                break
            lead += 1
        ever_ok = any(ok for ok, _ in trials)
        best_of_value = min((l for ok, l in trials if ok and l is not None), default=None)
        for n in THRESHOLDS:
            if lead >= n:
                per_n[n][0] += 1
                grand[n][0] += 1
                if ever_ok:
                    per_n[n][1] += 1
                    grand[n][1] += 1
                    killed_detail[n].append((cand, knob, val, lead, len(trials),
                                             best_of_value))
                    if (best_overall is not None and best_of_value is not None
                            and best_of_value <= best_overall * 1.02):
                        grand_best_killed[n].append((run.name, cand, knob, val, lead,
                                                     best_of_value, best_overall))
    for n in THRESHOLDS:
        tot, bad = per_n[n]
        if tot:
            print("   N=%-3d would retire %2d value(s); %d of them LATER SUCCEEDED"
                  % (n, tot, bad))
            for d in killed_detail[n][:4]:
                print("        killed: %s %s=%s  (lead-fails %d of %d trials, its best %s ms)"
                      % (d[0], d[1], d[2], d[3], d[4], d[5]))
    print()

print("########## ACROSS ALL RUNS")
for n in THRESHOLDS:
    tot, bad = grand[n]
    rate = (100.0 * bad / tot) if tot else 0.0
    print("  N=%-3d retire %3d values; %3d later succeeded  => over-block rate %.1f%%"
          % (n, tot, bad, rate))
print()
print("########## retired AND within 2% of the run's best (i.e. would have cost the result)")
if not any(grand_best_killed.values()):
    print("  none")
for n in THRESHOLDS:
    for row in grand_best_killed[n]:
        print("  N=%d: %s %s %s=%s lead=%d  its best %.4f vs run best %.4f"
              % (n, row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
