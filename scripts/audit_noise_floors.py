"""What is each task's measured ieee-vs-tf32 noise floor? The bounded-output premise failed.

Every rejection detail since the noise-floor line was added carries the reference's own
two-precision spread. Collect them per task rather than reasoning from output range.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FLOOR = re.compile(
    r"reference's OWN ieee-vs-tf32 spread[^:]*: \{([^}]*)\}")

per_task = defaultdict(list)
for run in sorted((REPO / "runs").glob("run-l3-*")):
    ev_path = run / "events.jsonl"
    if not ev_path.exists():
        continue
    task = run.name.split("-")[2]
    for line in ev_path.open(encoding="utf-8"):
        e = json.loads(line)
        # The floor line appears in BOTH sources. Reading only SPACE_REJECTED missed
        # L3:43 entirely, where the rejections that carry it are per-trial (bf16 fails
        # inside tuning rather than at witness time) -- see
        # docs/result-l3-43-bf16-rejected-above-the-floor.md.
        if e["type"] == "SPACE_REJECTED":
            detail = e["payload"].get("detail", "")
        elif e["type"] == "TRIAL_DONE":
            trial = e["payload"].get("trial") or e["payload"]
            detail = str(trial.get("failure_detail") or "")
        else:
            continue
        m = FLOOR.search(detail)
        if not m:
            continue
        body = m.group(1)
        frac = re.search(r"'frac_within_tol': ([0-9.]+)", body)
        cos = re.search(r"'cosine': '?([0-9.]+)'?", body)
        absmax = re.search(r"'ref_absmax': '([0-9.e+-]+)'", body)
        if frac:
            per_task[task].append((run.name, float(frac.group(1)),
                                   cos.group(1) if cos else "-",
                                   absmax.group(1) if absmax else "-"))

print(f"{'task':6}{'runs':6}{'n':>4}  {'floor frac (min..max)':26}{'ref_absmax':>12}")
for task in sorted(per_task):
    rows = per_task[task]
    fr = sorted(r[1] for r in rows)
    ams = {r[3] for r in rows}
    runs = len({r[0] for r in rows})
    print(f"L3:{task:3}{runs:6}{len(rows):>4}  "
          f"{fr[0]:.6f} .. {fr[-1]:.6f}      {sorted(ams)[0]:>12}")

print("\nGate threshold: 0.99")
for task in sorted(per_task):
    fr = [r[1] for r in per_task[task]]
    worst = min(fr)
    print(f"  L3:{task:4} floor {worst:.6f}  ->  gate is {0.99 - worst:+.4f} above the "
          f"floor  {'UNREACHABLE for a reassociation' if worst < 0.99 else 'reachable'}")
