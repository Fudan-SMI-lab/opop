"""Per-candidate check: does each precision branch get compensation/stabilization?
Plus wall-clock cost of trials by precision. Source + events, no GPU."""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])

# --- wall clock by precision, from consecutive TRIAL_DONE timestamps
prev_ts = None
cost = defaultdict(float)
count = defaultdict(int)
ok_count = defaultdict(int)
with open(run / "events.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") != "TRIAL_DONE":
            prev_ts = prev_ts if e.get("type") != "TUNING_DONE" else None
            continue
        ts = e.get("ts")
        tr = (e.get("payload") or {}).get("trial") or {}
        vals = (tr.get("params") or {}).get("values") or {}
        prec = vals.get("DOT_PRECISION") or vals.get("PREC") or "n/a"
        count[prec] += 1
        if (tr.get("status") or "").lower() == "complete":
            ok_count[prec] += 1
        if prev_ts is not None and isinstance(ts, (int, float)):
            dt = ts - prev_ts
            if 0 < dt < 600:      # ignore gaps that are agent calls, not trials
                cost[prec] += dt
        prev_ts = ts

print("=== wall clock attributable to each precision (sum of inter-trial gaps)")
tot = sum(cost.values())
for prec in sorted(cost, key=lambda k: -cost[k]):
    print("  %-6s %7.1f min  (%4.1f%% of trial time)  trials=%d  passed=%d"
          % (prec, cost[prec] / 60.0, 100 * cost[prec] / tot if tot else 0,
             count[prec], ok_count[prec]))
print("  TOTAL  %7.1f min" % (tot / 60.0))

# --- per-candidate: is each precision branch stabilized?
STAB = re.compile(r"_comp|rho|maximum\(tl\.max|split|/ *rho|scale")
print()
print("=== per-candidate precision branches: does the branch carry stabilization?")
cdir = run / "candidates"
if cdir.exists():
    for d in sorted(cdir.iterdir()):
        src = d / "source.py"
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        # find each DOT_PRECISION == "X" branch and scan to the next branch keyword
        marks = []
        for i, ln in enumerate(lines):
            m = re.search(r'DOT_PRECISION\s*==\s*[\'"](\w+)[\'"]', ln)
            if m:
                marks.append((i, m.group(1)))
            elif re.match(r"\s+else:\s*$", ln) and marks:
                marks.append((i, "else/fallback"))
        out = []
        for j, (i, name) in enumerate(marks):
            end = marks[j + 1][0] if j + 1 < len(marks) else min(i + 20, len(lines))
            body = "\n".join(lines[i:end])
            if "tl.dot" not in body:
                continue
            out.append("%s=%s" % (name, "STAB" if STAB.search(body) else "plain"))
        if out:
            print("  %-18s %s" % (d.name, "  ".join(out)))
