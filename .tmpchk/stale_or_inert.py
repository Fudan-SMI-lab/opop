"""a=2's metrics match a=1 to six decimals after a real source change. Inert, or stale?"""

import hashlib
import json
import re
from pathlib import Path

RUN = Path("runs/run-l3-21-20260905-071312")
ev = [json.loads(line) for line in (RUN / "events.jsonl").open(encoding="utf-8")]


def host(p: str) -> Path:
    return Path(p[5].upper() + ":/" + p[7:]) if p.startswith("/mnt/") else Path(p)


print("=== REPAIR_PRODUCED source_sha vs what the witness jobs actually ran ===")
for e in ev:
    if e["type"] == "REPAIR_PRODUCED":
        print(f"  repair @ {e['ts']:.0f}  input source_sha = "
              f"{e['payload'].get('source_sha', '?')[:16]}")

print()
jobs = sorted((f for f in (RUN / "jobs").glob("*-wit-*-eval-*.json")
               if not f.name.endswith(".out.json")),
              key=lambda p: p.stat().st_mtime)
for f in jobs:
    j = json.loads(f.read_text(encoding="utf-8"))
    s = host(j["kernel_src_path"])
    lab = f.name.split("-wit-")[1].split("-eval")[0]
    if not s.exists():
        print(f"  {lab:8} MISSING {s}")
        continue
    t = s.read_text(encoding="utf-8")
    sha = hashlib.sha256(t.encode()).hexdigest()[:16]
    # Does this source contain the a=2 "split high+residual" technique?
    split = bool(re.search(r"(hi|high)\b.*(lo|low|resid)", t, re.I)) and t.count("tl.dot(") >= 3
    dt = re.search(r"'COMPUTE_DTYPE':\s*'(\w+)'", t)
    print(f"  {lab:8} sha={sha}  bytes={len(t):6d}  dtype={dt.group(1) if dt else '-':5} "
          f"dots={t.count('tl.dot('):2}  split_technique={split}")

print("\n=== the witness sources for a=1 vs a=2, minimal label only ===")
mins = [f for f in jobs if "-wit-minimal-" in f.name]
srcs = []
for f in mins:
    j = json.loads(f.read_text(encoding="utf-8"))
    s = host(j["kernel_src_path"])
    if s.exists():
        srcs.append((f.name, hashlib.sha256(s.read_text(encoding="utf-8").encode()).hexdigest()))
for n, h in srcs:
    print(f"  {n}  {h[:16]}")
if len({h for _, h in srcs}) == 1 and len(srcs) > 1:
    print("  => IDENTICAL sources: the repair's change never reached the witness")
else:
    print("  => sources differ: the change reached the witness but did not move the metric")
