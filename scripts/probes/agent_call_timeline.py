import json
import os
import sys

rd = sys.argv[1]
d = sorted(x for x in os.listdir(rd) if x.startswith("run-"))[-1]
evs = [json.loads(l) for l in open(os.path.join(rd, d, "events.jsonl"), encoding="utf-8")
       if l.strip()]
for e in evs:
    if e["type"].startswith("AGENT_CALL"):
        p = e.get("payload") or {}
        extra = ""
        if e["type"] == "AGENT_CALL_FAILED":
            extra = " err=" + str(p.get("error"))[:120]
        print(f"  seq{e['seq']:4d} {e['type']:22s} {p.get('module','')}{extra}")
print(f"  candidates registered: {sum(1 for e in evs if e['type'] == 'CANDIDATE_REGISTERED')}")
print(f"  seeds spawned        : {sum(1 for e in evs if e['type'] == 'SEEDS_PRODUCED')}")
