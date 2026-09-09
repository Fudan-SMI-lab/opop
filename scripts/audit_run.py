"""Backend + precision + rescue audit for one run. Reads events.jsonl only."""
import json
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
ev = run / "events.jsonl"

backends = Counter()
detected = Counter()
mismatch = []
origins = Counter()
prec_by_status = Counter()
rescued = 0
screened = 0
fail_kinds = Counter()
near_ties = []
best = None

with open(ev, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "CANDIDATE_REGISTERED":
            c = p.get("candidate") or p
            b = c.get("backend")
            d = c.get("detected_backend") or c.get("backend_detected")
            backends[b] += 1
            detected[d] += 1
            if d and b and d != b:
                mismatch.append((c.get("candidate_id"), b, d))
            origins[c.get("origin")] += 1
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or p
            st = (tr.get("status") or "").lower()
            vals = (tr.get("params") or {}).get("values") or {}
            prec = vals.get("DOT_PRECISION") or vals.get("PREC") or "n/a"
            prec_by_status[(prec, st)] += 1
            if st != "complete":
                fail_kinds[tr.get("failure_kind")] += 1
            lat = tr.get("latency_ms") or {}
            m = lat.get("median")
            if st == "complete" and isinstance(m, (int, float)):
                if best is None or m < best:
                    best = m
            if tr.get("fp64_rescued") or tr.get("rescued_by_relative"):
                rescued += 1
        elif t == "CONFIG_SCREENED_INFEASIBLE":
            screened += 1

print("RUN:", run.name)
print("candidates by declared backend:", dict(backends))
print("candidates by detected backend:", dict(detected))
print("declared/detected MISMATCHES:", mismatch if mismatch else "none")
print("candidates by origin:", dict(origins))
print("screened infeasible (pre-trial):", screened)
print("best tuned median:", best)
print("--- trials by precision x status ---")
for (prec, st), n in sorted(prec_by_status.items()):
    print("  %-8s %-24s %d" % (prec, st, n))
print("--- failure kinds ---")
for k, n in fail_kinds.most_common():
    print("  %-32s %d" % (k, n))
