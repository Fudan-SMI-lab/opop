"""After the real filters, does the shipped rule request MORE than the pre-fix rule did?

`audit_fallback_vs_prefix_rule.py` shows the median fallback (`_median_direction`) is wider than
the pre-fix `_param_stat` flag on 219 knobs and narrower on none -- it checks only
"argmin sits on the measured edge" and drops the monotone-tail and beyond-is-blocked tests.

That is a per-knob comparison of the raw predicates. It is NOT yet a claim about behaviour,
because `boundary_knobs_to_expand` then applies filters that reject many knobs regardless of
where the flag came from: numeric-only, `effect_pct >= min_effect_pct`, and the hard-edge wall
list (NUM_WARPS=1, BLOCK_K=16, ...). This asks the question that matters: after those filters,
does the shipped two-pass rule ask for more knobs than the pre-fix rule would have?

If yes, the fix's "floor, not veto" framing understates it -- the fallback would be widening
expansion on spaces where the old code requested nothing, i.e. spending expansions the project
never spent before.

Read-only; safe against a live run.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand  # noqa: E402
from kernel_optimizer.models.core import ParameterSpace  # noqa: E402
from kernel_optimizer.models.reports import ParamStat, TuningStats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_fallback_vs_prefix_rule import old_rule  # noqa: E402


def prefix_requests(st: TuningStats, sp: ParameterSpace | None) -> list[dict]:
    """What `boundary_knobs_to_expand` would return if the flags carried the PRE-FIX aim.

    Re-stamps each ParamStat's at_boundary/boundary_direction from `old_rule`, then runs the
    real predicate -- so the filters are identical and only the aim differs.
    """
    patched = st.model_copy(deep=True)
    for ps in patched.param_stats:
        flag, direction = old_rule(ps, sp)
        ps.at_boundary = flag
        ps.boundary_direction = direction
        # Force the anchored pass to be the only one that can fire, by making the winner
        # anchor agree with the median: otherwise the shipped fallback would run too.
        ps.best_trial_value = None
    return boundary_knobs_to_expand(patched, idle_frac=1.0, space=sp)


runs = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/run-l3-*"))
n_spaces = 0
same = wider = narrower = 0
wider_examples: list[str] = []
newly_requesting = 0  # spaces the pre-fix rule left with NO request at all
for run in runs:
    ev = Path(run) / "events.jsonl"
    if not ev.exists():
        continue
    spaces: dict[str, ParameterSpace] = {}
    for line in open(ev, encoding="utf-8"):
        e = json.loads(line)
        if e["type"] == "SPACE_PUBLISHED":
            sp = ParameterSpace.model_validate(e["payload"]["space"])
            spaces[sp.space_id] = sp
        elif e["type"] == "STATS_DONE":
            st = TuningStats.model_validate(e["payload"]["stats"])
            sp = spaces.get(st.space_id)
            if sp is None:
                continue
            n_spaces += 1
            shipped = {(k["name"], k["direction"]) for k in
                       boundary_knobs_to_expand(st, idle_frac=1.0, space=sp)}
            old = {(k["name"], k["direction"]) for k in prefix_requests(st, sp)}
            if shipped == old:
                same += 1
            elif shipped > old or (shipped - old and not (old - shipped)):
                wider += 1
                if not old:
                    newly_requesting += 1
                if len(wider_examples) < 10:
                    wider_examples.append(
                        f"{Path(run).name} {st.space_id}: old={sorted(old)} "
                        f"shipped={sorted(shipped)}"
                    )
            else:
                narrower += 1

print(f"spaces replayed: {n_spaces}")
print(f"  identical request set            {same}")
print(f"  shipped requests MORE            {wider}   "
      f"(of which {newly_requesting} had no pre-fix request at all)")
print(f"  shipped requests fewer/different {narrower}")
print("\nexamples where shipped asks for more:")
for line in wider_examples:
    print("  " + line)
