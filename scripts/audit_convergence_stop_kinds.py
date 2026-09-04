"""Has stop_kind='converged' EVER fired, on any run? The window may be off by one.

best_history starts empty and gains one entry per completed rewrite round; the seed's
latency is never in it. family_verdict needs len(history) >= no_improve_rounds + 1 to judge
convergence, i.e. 3 entries for no_improve_rounds=2 -- but rewrite_rounds_per_family=3
freezes on budget at used=3, and the check runs BEFORE the third round is recorded. So the
converged branch may be structurally unreachable at these settings.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

print(f"{'run':34}{'family':14}{'verdict':10}{'stop_kind':20}{'history'}")
kinds = {}
for run in sorted((REPO / "runs").glob("run-l3-*")):
    ev_path = run / "events.jsonl"
    if not ev_path.exists():
        continue
    for line in ev_path.open(encoding="utf-8"):
        e = json.loads(line)
        if e["type"] != "CONVERGENCE_DECIDED":
            continue
        d = e["payload"]["decision"]
        if d["scope"] != "family" or d["verdict"] != "freeze":
            continue
        sk = d.get("stop_kind")
        kinds[sk] = kinds.get(sk, 0) + 1
        print(f"{run.name:34}{e['payload'].get('family_id', '?'):14}"
              f"{d['verdict']:10}{str(sk):20}"
              f"{d['evidence'].get('best_history')}")

print()
print("family freeze stop_kinds across every L3 run:")
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
if "converged" not in kinds:
    print("\n  'converged' has NEVER fired on any L3 run.")
