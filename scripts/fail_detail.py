"""Print raw failure_detail for N trials of a given precision, unredacted. events.jsonl only."""
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
want = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3

shown = 0
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
        if (tr.get("status") or "").lower() == "complete":
            continue
        if tr.get("failure_kind") != "correctness_mismatch":
            continue
        vals = (tr.get("params") or {}).get("values") or {}
        prec = vals.get("DOT_PRECISION") or vals.get("PREC") or "n/a"
        if prec != want:
            continue
        print("=== cand=%s params=%s" % (tr.get("candidate_id"), json.dumps(vals)))
        print(tr.get("failure_detail") or "(no detail)")
        print()
        shown += 1
        if shown >= limit:
            break
if shown == 0:
    print("no correctness_mismatch trials for precision", want)
