"""Audit: did `eval_semantics.md` actually reach the agents, and did it stop the BN failures?

The L3:21 reference runs in TRAIN mode, so BatchNorm must use batch statistics rather than
`running_mean`/`running_var`. Candidates kept getting this wrong until the probe
(`SEMANTICS_PROBED` -> `task/eval_semantics.md`) started telling them.

Two failure signatures are counted, both readable from the candidate source:
  - a runtime `if ....training:` branch, which strands the optimization on the dead side of a
    branch that is never taken (correctness passes, timing looks normal, every trial measures
    the fallback);
  - any `running_mean` / `running_var` reference, which is the train/eval mismatch itself.

The useful output is not the per-run trend (dates confound it) but the per-ORIGIN split within
a single run: on run-l3-21-20260904-013056 the generator had the doc and the rewriter did not,
so seeds and rewrites in the same run form a controlled comparison.

Usage:  python scripts/audit_eval_semantics_reach.py [--runs runs] [--task-glob 'run-l3-21-*']
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

TRAINING_BRANCH = re.compile(r"if\s+[\w.]*\.training\s*:")
RUNNING_STATS = re.compile(r"running_mean|running_var")


def signatures(source: str) -> list[str]:
    hits = []
    if TRAINING_BRANCH.search(source):
        hits.append("training-branch")
    if RUNNING_STATS.search(source):
        hits.append("running-stats")
    return hits


def load_events(run: Path) -> list[dict]:
    f = run / "events.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--task-glob", default="run-l3-21-*")
    args = ap.parse_args()

    print(f"{'run':14s} {'probe':>6s} {'cands':>6s} {'affected':>9s}   "
          f"sandboxes with eval_semantics.md, by module")
    per_origin_totals: Counter = Counter()
    per_origin_affected: Counter = Counter()

    for d in sorted(Path(args.runs).glob(args.task_glob)):
        cdir = d / "candidates"
        if not cdir.exists():
            continue
        events = load_events(d)
        probed = sum(1 for e in events if e.get("type") == "SEMANTICS_PROBED")

        origin = {}
        for e in events:
            if e.get("type") == "CANDIDATE_REGISTERED":
                c = (e.get("payload") or {}).get("candidate") or e.get("payload") or {}
                if c.get("candidate_id"):
                    origin[c["candidate_id"]] = c.get("origin") or "?"

        total = affected = 0
        by_origin_t: Counter = Counter()
        by_origin_a: Counter = Counter()
        for c in sorted(cdir.iterdir()):
            src = c / "source.py"
            if not src.exists():
                continue
            hits = signatures(src.read_text(encoding="utf-8"))
            o = origin.get(c.name, "?")
            total += 1
            by_origin_t[o] += 1
            per_origin_totals[o] += 1
            if hits:
                affected += 1
                by_origin_a[o] += 1
                per_origin_affected[o] += 1

        # Which modules' sandboxes actually got the doc.
        sbs = sorted((d / "sandboxes").iterdir()) if (d / "sandboxes").exists() else []
        mods: Counter = Counter()
        have: Counter = Counter()
        for s in sbs:
            m = s.name.rsplit("-", 1)[0]
            mods[m] += 1
            if (s / "task" / "eval_semantics.md").exists():
                have[m] += 1
        reach = " ".join(f"{m}:{have.get(m, 0)}/{mods[m]}" for m in sorted(mods))

        print(f"{d.name[-13:]:14s} {probed:6d} {total:6d} {affected:9d}   {reach}")
        if affected and len(by_origin_t) > 1:
            split = "  ".join(f"{o}: {by_origin_a[o]}/{by_origin_t[o]}"
                              for o in sorted(by_origin_t))
            print(f"{'':14s} {'':6s} {'':6s} {'':9s}   by origin -> {split}")

    print("\nacross all matched runs, affected by origin:")
    for o in sorted(per_origin_totals):
        t = per_origin_totals[o]
        a = per_origin_affected[o]
        print(f"   {o:10s} {a:3d} / {t:3d}  ({a / t * 100:.0f}%)" if t else f"   {o}: 0")
    print("\nThe per-run trend is confounded by date (prompt and driver changes landed between")
    print("runs). The per-ORIGIN split inside one run is the controlled comparison: on")
    print("run-l3-21-20260904-013056 the generator had the doc (seeds 0/4 affected) and the")
    print("rewriter did not (rewrites 9/12).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
