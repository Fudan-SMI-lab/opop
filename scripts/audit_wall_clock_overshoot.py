"""By how much do runs exceed `wall_clock_hours`, and is the cap a cap or a floor?

`global_verdict` compares elapsed hours against `wall_clock_hours` and freezes when it is
reached -- but it is consulted ONLY at the top of each outer round (orchestrator.py:427).
Once a round starts it runs to completion: a rewrite round over two active families, each
with a parameterizer call, a 40-trial tune, possible expansion and re-tune, plus an analyst
call. So the budget behaves as a "do not START past this point" floor, not a hard ceiling.

Observed live: run-l3-21-20260905-195615 passed 12.00h at 07:55 and was still tuning new
candidates at 08:37 (12.70h), because the round that began at 05:53 had not returned.

Measures the overshoot across every run that hit the budget, so the size of the effect is a
number rather than an impression. No GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CAP = 12.0  # budgets.wall_clock_hours in configs/experiments_l3.yaml

rows = []
for run in sorted(Path("runs").glob("run-l3-*")):
    f = run / "events.jsonl"
    if not f.exists():
        continue
    evs = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            evs.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    if not evs:
        continue
    t0 = evs[0]["ts"]
    total = (evs[-1]["ts"] - t0) / 3600
    # the last global verdict that said continue, and when the budget was first crossed
    last_continue = None
    crossed = None
    stop_kind = None
    for e in evs:
        p = e.get("payload", {})
        if e["type"] == "CONVERGENCE_DECIDED":
            d = p.get("decision", {})
            if d.get("scope") != "global":
                continue
            el = d.get("evidence", {}).get("elapsed_hours")
            if d.get("verdict") == "continue":
                last_continue = el
            else:
                stop_kind = d.get("stop_kind")
        if crossed is None and (e["ts"] - t0) / 3600 >= CAP:
            crossed = (e["ts"] - t0) / 3600
    rows.append({"run": run.name, "total_h": total, "last_continue": last_continue,
                 "crossed": crossed, "stop_kind": stop_kind,
                 "finished": any(e["type"] == "RUN_FINISHED" for e in evs)})

over = [r for r in rows if r["crossed"] is not None]
print(f"runs on disk: {len(rows)}   runs that reached the {CAP}h budget: {len(over)}")
if not over:
    print("\nNo run has ever hit the wall clock, so the overshoot has never mattered in")
    print("practice -- every completed run froze on family budgets instead. The check's")
    print("round granularity is therefore a latent property, not an observed cost.")
else:
    print()
    for r in over:
        ov = r["total_h"] - CAP
        print(f"  {r['run']:34s} total {r['total_h']:6.2f}h  overshoot {ov:+.2f}h"
              f"  last 'continue' at {r['last_continue']}h  stop={r['stop_kind']}"
              f"  finished={r['finished']}")
    ovs = [r["total_h"] - CAP for r in over]
    print(f"\novershoot: median {statistics.median(ovs):+.2f}h  max {max(ovs):+.2f}h")

print("""
Interpretation. Checking at round granularity is a deliberate trade, not an oversight: an
outer round is the unit that produces a comparable family result, and killing one mid-flight
would spend the GPU time and journal nothing usable. The cost is that a run can exceed its
stated budget by up to one round -- and a round here has been measured at 30-60 min, with
individual trials taking 30 min when compilation is slow.

Two ways this could bite that are worth stating:
  * a chain schedule built on `wall_clock_hours` will under-estimate: the next arm starts
    late by the overshoot;
  * a run whose LAST round is unusually slow overshoots most, i.e. the error is largest
    exactly when the budget matters most.

Not proposed as a change here. A mid-round budget check would need somewhere safe to cut
(after a candidate completes, not mid-tune) and that is a behaviour change to the outer loop
affecting every run. Recorded so the chain estimator's numbers are read as floors.
""")
