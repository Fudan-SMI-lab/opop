"""One arm's final numbers, with the raw sample arrays stripped out.

`RUN_FINISHED` carries the baseline's full 100-sample array (and more), so dumping the payload prints
hundreds of floats and buries the six fields that matter. This walks the payload and elides any list
of numbers longer than a handful, printing its length and range instead -- the samples are evidence
worth keeping in the log, just not worth reading here.
"""
import json
import os
import sys


def strip(obj, depth=0):
    if isinstance(obj, dict):
        return {k: strip(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        nums = [x for x in obj if isinstance(x, (int, float))]
        if len(obj) > 6 and len(nums) == len(obj):
            return (f"<{len(obj)} samples: min {min(nums):.4f} median "
                    f"{sorted(nums)[len(nums) // 2]:.4f} max {max(nums):.4f}>")
        return [strip(x, depth + 1) for x in obj]
    return obj


rd = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "RUN_FINISHED"
found = None
with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
    for ln in fh:
        if want not in ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("type") == want:
            found = e
if found is None:
    print(f"no {want} in {rd}")
    raise SystemExit(1)
print(json.dumps(strip(found.get("payload") or {}), indent=2, ensure_ascii=False))
