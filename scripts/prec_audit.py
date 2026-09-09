"""fp64-relative-arm ratio and trial ORDER by precision. Answers: is a precision genuinely
less accurate, and does the sampler keep spending on a precision that is 0-for-N?"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

run = Path(sys.argv[1])

ratios = defaultdict(list)
order = []          # (seq, precision, status) in event order
per_cand_prec = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cand -> prec -> [ok, fail]

RAT = re.compile(r"'ratio_to_reference': '([0-9.]+|inf)'")
MULT = re.compile(r"'multiplier': ([0-9.]+)")
mults = defaultdict(set)

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
        vals = (tr.get("params") or {}).get("values") or {}
        prec = vals.get("DOT_PRECISION") or vals.get("PREC") or "n/a"
        cand = tr.get("candidate_id")
        order.append((e.get("seq"), prec, st))
        slot = per_cand_prec[cand][prec]
        if st == "complete":
            slot[0] += 1
        else:
            slot[1] += 1
        d = tr.get("failure_detail") or ""
        m = RAT.search(d)
        if m and m.group(1) != "inf":
            ratios[prec].append(float(m.group(1)))
        mm = MULT.search(d)
        if mm:
            mults[prec].add(float(mm.group(1)))

print("=== fp64-relative-arm ratio_to_reference by precision (FAILED trials only)")
for prec in sorted(ratios):
    v = sorted(ratios[prec])
    n = len(v)
    print("  %-6s n=%-4d min=%.3f  median=%.3f  max=%.3f   multiplier(s) seen=%s"
          % (prec, n, v[0], v[n // 2], v[-1], sorted(mults.get(prec, set()))))

print()
print("=== does the sampler keep spending on a hopeless precision?")
# split trial order into quarters, count each precision's share and pass rate
q = max(1, len(order) // 4)
for qi in range(4):
    chunk = order[qi * q:(qi + 1) * q] if qi < 3 else order[3 * q:]
    tot = len(chunk)
    by = defaultdict(lambda: [0, 0])
    for _, prec, st in chunk:
        by[prec][0 if st == "complete" else 1] += 1
    parts = []
    for prec in sorted(by):
        ok, bad = by[prec]
        parts.append("%s %d/%d ok" % (prec, ok, ok + bad))
    print("  quarter %d (n=%d): %s" % (qi + 1, tot, "; ".join(parts)))

print()
print("=== per-candidate x precision (ok/total) -- is any precision 0-for-N on EVERY candidate?")
for cand in sorted(per_cand_prec):
    parts = []
    for prec in sorted(per_cand_prec[cand]):
        ok, bad = per_cand_prec[cand][prec]
        parts.append("%s=%d/%d" % (prec, ok, ok + bad))
    print("  %-18s %s" % (cand, "  ".join(parts)))
