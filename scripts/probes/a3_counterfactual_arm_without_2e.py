"""The A3 counterfactual: did an arm WITHOUT 2e free its own shared-memory walls anyway?

WHY ARM 2 IS NOT A CONTROL THE OBVIOUS WAY. 2e is switched off in arm 2, so it emits no
RESOURCE_WALL_ATTRIBUTED events at all -- asking it for attributed walls returns 0 by construction,
which says nothing. A3's real counterfactual is: arm 2 hit shared-memory refusals too (its own log
records them), so did ITS rewrites free those dimensions without ever being told about them?

If they did, arm 3's positive A3 is not evidence for the steer -- the rewriter would be relieving
shared memory on its own, and 2e's prompt fragment would be decoration. If they did not, the steer is
doing something.

HOW THIS AVOIDS INVENTING A SECOND WALL-FINDER. It calls the SHIPPED `find_walls` on the
`TuningStats` that arm 2 already serialized into `STATS_DONE`, plus the refused parameter sets from
its own `TRIAL_DONE` records with `failure_kind == "infeasible_shared_memory"` -- exactly the two
inputs `orchestrator._attribute_resource_walls` passes (orchestrator.py:1472-1474). A reimplementation
here would be free to disagree with the one the arm-3 numbers came from, and then the comparison would
be between two readers rather than between two arms.

No GPU: `find_walls` is pure, and the probing step (which does compile) is deliberately NOT run --
this asks only "was a dimension truncated, and did a later rewrite un-truncate it", which the log
answers. That means these walls are FOUND, not ATTRIBUTED: attribution needs the compile probe. The
comparison is therefore between arm 3's found walls and arm 2's found walls, and the script prints
arm 3's found count too so the two are read at the same level.
Usage: a3_counterfactual_arm_without_2e.py <checkout-src-dir> label=<run_dir> [label=<run_dir> ...]

The checkout path is an ARGUMENT, not a constant. A hardcoded guess resolved to a stale sibling
checkout on box 4 (`opop-workspace/opop`, which predates this module) and the import failed -- the
good outcome, but the same guess landing on an OLDER copy of `wall_attribution.py` would have run a
different wall-finder than the one that produced arm 3's numbers and reported the difference as an
arm difference. Pass the `src` of the checkout the arms actually ran from; find it with
`readlink -f /proc/<pid>/cwd` and confirm its commit with `git log --oneline -1`.
"""
import json
import os
import sys
from collections import defaultdict

if len(sys.argv) < 3 or "=" in sys.argv[1]:
    print(__doc__)
    raise SystemExit(2)
_SRC = sys.argv[1]
sys.path.insert(0, _SRC)
try:
    from kernel_optimizer.evaluation import wall_attribution
    from kernel_optimizer.models.reports import TuningStats
except ImportError as exc:  # pragma: no cover
    print(f"cannot import the shipped wall finder from {_SRC}: {exc}")
    print("This script deliberately does NOT reimplement find_walls -- a second implementation")
    print("could disagree with the one that produced the numbers being compared.")
    raise SystemExit(2)
print(f"wall finder: {wall_attribution.__file__}")


def load(rd):
    out = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def analyse(rd, label):
    evs = load(rd)
    children = defaultdict(list)
    for e in evs:
        if e.get("type") != "CANDIDATE_REGISTERED":
            continue
        c = (e.get("payload") or {}).get("candidate") or {}
        for p in (c.get("parent_ids") or []):
            if c.get("candidate_id"):
                children[p].append(c["candidate_id"])

    trials = defaultdict(list)
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("candidate_id"):
            trials[t["candidate_id"]].append(t)

    # STATS_DONE carries the whole TuningStats, but not the candidate id -- it is attributed by the
    # STATS_DONE that follows each candidate's TUNING_DONE. Pair them in log order.
    stats_for = {}
    pending = None
    for e in evs:
        if e.get("type") == "TUNING_DONE":
            pending = (e.get("payload") or {}).get("candidate_id")
        elif e.get("type") == "STATS_DONE" and pending:
            raw = (e.get("payload") or {}).get("stats")
            if raw:
                try:
                    stats_for[pending] = TuningStats.model_validate(raw)
                except Exception:
                    pass
            pending = None

    def eq(a, b):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return a == b

    def descend(cid):
        out, stack = [], list(children.get(cid, []))
        while stack:
            x = stack.pop(0)
            out.append(x)
            stack.extend(children.get(x, []))
        return out

    print(f"=== {label} :: {os.path.basename(rd)}")
    total_found = 0
    freed = walled = gone = untried = 0
    for cid, stats in stats_for.items():
        refused = [(t.get("params") or {}).get("values") or {} for t in trials.get(cid, [])
                   if t.get("failure_kind") == "infeasible_shared_memory"]
        if not refused:
            continue
        walls = wall_attribution.find_walls(stats, refused)
        if not walls:
            continue
        for w in walls:
            total_found += 1
            knob, V = w.param, w.refused_value
            kids = descend(cid)
            outcome = "NO DESCENDANT"
            for k in kids:
                ok = any(t.get("status") == "complete"
                         and knob in ((t.get("params") or {}).get("values") or {})
                         and eq(((t.get("params") or {}).get("values") or {})[knob], V)
                         for t in trials.get(k, []))
                still = any(t.get("failure_kind") == "infeasible_shared_memory"
                            and knob in ((t.get("params") or {}).get("values") or {})
                            and eq(((t.get("params") or {}).get("values") or {})[knob], V)
                            for t in trials.get(k, []))
                has_knob = any(knob in ((t.get("params") or {}).get("values") or {})
                               for t in trials.get(k, []))
                if ok:
                    outcome = "FREED"
                    break
                if still:
                    outcome = "STILL WALLED"
                elif not has_knob and outcome == "NO DESCENDANT":
                    outcome = "DIMENSION GONE"
                elif outcome == "NO DESCENDANT":
                    outcome = "NOT TRIED"
            freed += outcome == "FREED"
            walled += outcome == "STILL WALLED"
            gone += outcome == "DIMENSION GONE"
            untried += outcome in ("NOT TRIED", "NO DESCENDANT")
            print(f"  {cid} {knob}: ran {w.ran_values} refused {V} "
                  f"monotone={w.monotone} tail={w.tail_gain_pct:+.1f}%  ==> {outcome}")
    print(f"  walls FOUND {total_found}: freed {freed}, still walled {walled}, "
          f"dimension gone {gone}, not tried {untried}")
    print()


for spec in sys.argv[2:]:
    label, rd = spec.split("=", 1)
    analyse(rd, label)
print("READING: arm 3 was TOLD about its walls; arm 2 was not. If arm 2's walls get freed at the")
print("same rate, the 2e steer is not what freed arm 3's -- the rewriter relieves shared memory")
print("anyway. These are FOUND walls on both sides (attribution needs the compile probe), so the")
print("two arms are compared at the same level.")
