"""What knob names carry a precision choice, and what are the pass rates per (knob,value)?
Generalizes prec_audit across runs whose knob is not called DOT_PRECISION."""
import json
import sys
from collections import defaultdict
from pathlib import Path

PREC_VALUES = {"fp16", "float16", "half", "bf16", "bfloat16", "tf32", "ieee",
               "fp32", "float32", "fp64", "float64"}

runs = [Path(a) for a in sys.argv[1:]]
for run in runs:
    ev = run / "events.jsonl"
    if not ev.exists():
        print("%s: no events.jsonl" % run.name)
        continue
    # knob -> value -> [ok, fail]
    tally = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    knobs_seen = set()
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
            st = (tr.get("status") or "").lower()
            vals = (tr.get("params") or {}).get("values") or {}
            for k, v in vals.items():
                knobs_seen.add(k)
                if isinstance(v, str) and v.lower() in PREC_VALUES:
                    slot = tally[k][v]
                    slot[0 if st == "complete" else 1] += 1
    print("=== %s" % run.name)
    if not tally:
        print("   NO precision-valued knob in any trial. knobs seen: %s"
              % sorted(knobs_seen))
    for k in sorted(tally):
        parts = []
        for v in sorted(tally[k]):
            ok, bad = tally[k][v]
            parts.append("%s %d/%d" % (v, ok, ok + bad))
        print("   %-16s %s" % (k, "  ".join(parts)))
    print()
