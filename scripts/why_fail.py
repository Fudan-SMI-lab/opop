"""Why did a given precision fail? Group failure detail by precision. events.jsonl only."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
want = sys.argv[2] if len(sys.argv) > 2 else None

by_prec = defaultdict(Counter)
details = defaultdict(list)
cands = defaultdict(Counter)

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
            continue
        tr = (e.get("payload") or {}).get("trial") or {}
        st = (tr.get("status") or "").lower()
        if st == "complete":
            continue
        vals = (tr.get("params") or {}).get("values") or {}
        prec = vals.get("DOT_PRECISION") or vals.get("PREC") or "n/a"
        if want and prec != want:
            continue
        fk = tr.get("failure_kind") or "?"
        by_prec[prec][fk] += 1
        cands[prec][tr.get("candidate_id")] += 1
        d = (tr.get("failure_detail") or "").strip()
        if d:
            # normalize numbers so distinct messages collapse
            norm = re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", "N", d)[:220]
            details[prec].append(norm)

for prec in sorted(by_prec):
    print("=== precision %s: %d failures" % (prec, sum(by_prec[prec].values())))
    for fk, n in by_prec[prec].most_common():
        print("   kind %-30s %d" % (fk, n))
    print("   across candidates:", dict(cands[prec]))
    top = Counter(details[prec]).most_common(4)
    for msg, n in top:
        print("   [%3d] %s" % (n, msg))
    print()
