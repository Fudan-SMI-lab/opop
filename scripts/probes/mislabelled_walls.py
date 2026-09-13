"""How often does a shared-memory refusal reach the trial and get MISLABELLED `runtime_error`?

THE DEFECT. `worker_main._classify_exception` maps every non-OOM exception to `runtime_error`. Triton's
launch-time refusal raises `triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: N, Hardware limit: M` -- a refusal, not a crash -- so it lands as `runtime_error`.

WHY IT MATTERS HERE SPECIFICALLY. `find_walls`' entire input is the set of trials with
`failure_kind == "infeasible_shared_memory"` (orchestrator.py:1573). A refusal that arrives under the
wrong label is invisible to C2 and to S7 -- the mechanism this pair is measuring. It also crosses two
other consumers: `tpe.py:218` treats `infeasible_shared_memory` as a HARD failure and `runtime_error`
as a soft one, and `deweight.py` pools `runtime_error` as evidence of a whole-candidate defect, which a
per-configuration resource refusal is not.

WHY IT IS NORMALLY INVISIBLE. `compile_screen` catches these before launch and records the right label,
which is why a healthy run shows a dozen `infeasible_shared_memory` and no such `runtime_error`. The
mislabel needs the SCREEN to fail first -- and a screen failure is deliberately never a verdict
(`correctness.py`), so the real trial then runs and hits the runtime. On the live S7 treatment arm that
happened when `ptxas` blew past `screen_timeout_s` on an 8 MB / 161,770-line PTX.

WHAT THIS DECIDES. Frequency, and nothing else. A defect that fires once per run is inside the noise
floor and the fix belongs in the next round; one that fires often enough to shift `find_walls`' input is
a validity problem on a pair that is measuring exactly that input, and the standing rule is to fix and
restart rather than to finish and caveat. The count is per-arm and per-run because the two arms are
independent searches.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/mislabelled_walls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")

# The runtime refusal's own words, lowercased. Both spellings are checked because the exception's
# str() carries the message while its type name carries the class -- and `_classify_exception` sees
# "TypeName: message", so either alone would be a partial test.
_MARKS = ("out of resource: shared memory", "outofresources")


def _detail(t: dict) -> str:
    # `failure_detail` is the field name. `failure_message` does not exist on TrialRecord -- reading
    # it returned "" for a trial whose traceback was in fact 1150 chars long, which is the
    # a-constant-reading-is-a-broken-probe shape: a plausible empty string, not a None.
    return str(t.get("failure_detail") or "").lower()


def scan(run: Path) -> dict:
    kinds: dict[str, int] = {}
    mislabelled: list[dict] = []
    correctly: int = 0
    screen_note = 0
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        kind = t.get("failure_kind") or "ok"
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "infeasible_shared_memory":
            correctly += 1
            continue
        d = _detail(t)
        if any(m in d for m in _MARKS):
            mislabelled.append({
                "trial": t.get("trial_id"), "kind": kind,
                "wall_s": t.get("job_wall_s"),
                "params": (t.get("params") or {}).get("values") or {},
            })
        elif kind not in ("ok",) and "screen" in d:
            screen_note += 1
    n = sum(kinds.values())
    return {"run": run.name, "trials": n, "kinds": kinds,
            "correctly_labelled": correctly, "mislabelled": mislabelled,
            "screen_mentions": screen_note}


def main() -> int:
    runs = sorted(p.parent for p in BASE.glob("*/run-*/events.jsonl"))
    if not runs:
        print("no runs under %s" % BASE)
        return 2

    tot_trials = tot_ok = tot_bad = 0
    hit_runs = 0
    print("%-46s %7s %7s %7s" % ("run", "trials", "correct", "MISLBL"))
    for r in runs:
        s = scan(r)
        tot_trials += s["trials"]
        tot_ok += s["correctly_labelled"]
        tot_bad += len(s["mislabelled"])
        if s["mislabelled"]:
            hit_runs += 1
        print("%-46s %7d %7d %7d" % (
            "%s/%s" % (r.parent.name, r.name)[-46:], s["trials"],
            s["correctly_labelled"], len(s["mislabelled"])))
        for m in s["mislabelled"]:
            keep = {k: m["params"].get(k) for k in ("BLOCK_M", "BLOCK_N", "NUM_WARPS",
                                                    "NUM_STAGES", "COMPUTE_DTYPE", "DOT_MODE")
                    if k in m["params"]}
            print("      -> %s  as %-14s wall %6.0f s  %s" % (
                m["trial"], m["kind"], m["wall_s"] or 0, keep))

    print()
    print("TOTAL trials %d   correctly labelled refusals %d   MISLABELLED %d" % (
        tot_trials, tot_ok, tot_bad))
    if tot_ok + tot_bad:
        print("  the mislabelled share of ALL shared-memory refusals: %.1f%%  (%d of %d)" % (
            100.0 * tot_bad / (tot_ok + tot_bad), tot_bad, tot_ok + tot_bad))
    print("  runs with at least one: %d of %d" % (hit_runs, len(runs)))
    print()
    print("HOW TO READ IT. The denominator that matters is not the trial count -- it is the refusal")
    print("count, because the refusals ARE find_walls' input. A few percent of a wall search that")
    print("already only covers a quarter of families is a next-round fix; a large share would mean the")
    print("live pair is measuring a degraded version of its own mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
