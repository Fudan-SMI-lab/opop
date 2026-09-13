"""Print one real TRIAL_DONE payload verbatim. No filtering, no assumptions about field values.

Written because a probe that guessed `status == "ok"` and `profile.compile_s` returned n=0 on all
three arms -- which is the good failure (loud, immediate) but still a probe I would have had to
distrust if it had returned plausible numbers instead. Read the emitter's actual output before
filtering on it.
"""
import json
import os
import sys
from collections import Counter

rd = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
statuses = Counter()
shown = 0
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        if "TRIAL_DONE" not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        p = e.get("payload") or {}
        t = p.get("trial")
        if t is None:
            if shown < n:
                print("!! payload has no 'trial' key. payload keys:", sorted(p.keys()))
                print(json.dumps(p, indent=2, ensure_ascii=False)[:3000])
                shown += 1
            continue
        statuses[t.get("status")] += 1
        if shown < n:
            print("payload keys:", sorted(p.keys()))
            print(json.dumps(t, indent=2, ensure_ascii=False)[:3000])
            shown += 1
print()
print("status values:", dict(statuses))
