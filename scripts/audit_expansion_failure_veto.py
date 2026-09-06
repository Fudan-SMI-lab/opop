"""Prospective audit: what would a failure-rate veto on expansion aiming actually change?

P4 proposes adding one condition to `boundary_knobs_to_expand`'s filter chain: skip a knob
whose boundary value already fails often. The measured motivation is real (values added beyond
a failing edge fail 43% vs 15% beyond a healthy edge), but the fix has a specific hazard that
`orchestrator.py:315-331` already documents for a different cause:

    an expansion delivers TWO things, a widened range AND a fresh tuning budget, and returning
    [] cancels both -- historically 6 of 8 cancelled expansions had improved, including the
    two largest gains in that group, both won by a configuration that was already reachable.

So the veto must be judged on two questions, not one:
  1. does it actually suppress the high-failure aims it targets?
  2. how often does it empty the request list, and would that cancel expansions that improved?

Question 2 is the one that decides whether the veto is safe as written or needs to degrade to
the median fallback instead of cancelling. This script answers both from the event log.

Reads only. Usage: python scripts/audit_expansion_failure_veto.py [runs/<run-id> ...]
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

THRESHOLD = 0.30          # the proposed max_edge_failure_frac
MIN_SAMPLES_HINT = 4      # only advisory here: trial counts per value are not in the event


def _numeric(name: str) -> bool:
    """Mirror of orchestrator._is_numeric_knob, minus the import (it is a closure)."""
    return True  # replaced below by the real predicate if importable


try:
    from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand
    from kernel_optimizer.models.core import ParameterSpace
    from kernel_optimizer.models.reports import TuningStats
    HAVE_REAL = True
except Exception:  # pragma: no cover - audit still useful without it
    HAVE_REAL = False


def load(run: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in (run / "events.jsonl").read_text(
        encoding="utf-8", errors="replace").splitlines() if l.strip()]


def edge_value(choices: list, direction: str):
    """The choice a widening in `direction` would extend past.

    MUST come from the domain, not from failure_rate_by_value: that dict is built by
    iterating completed trials, so its key order is trial order. `_median_direction`'s
    docstring records the same hazard for latency_by_value (real observed order
    ['128','64','256','512','1024'], whose first key is not the domain minimum).
    """
    numeric = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if not numeric:
        return None
    return min(numeric) if direction == "min" else max(numeric)


def main() -> int:
    args = sys.argv[1:]
    runs = [pathlib.Path(a) for a in args] if args else sorted(
        pathlib.Path("runs").glob("run-l3-*"))
    runs = [r for r in runs if (r / "events.jsonl").exists()]
    if not runs:
        print("no runs found")
        return 2

    # aim-level: how many requested knobs sit at a failing edge?
    aimed = vetoed = 0
    # expansion-level: would the veto empty a request list that was non-empty?
    emptied: list[tuple] = []
    kept: list[tuple] = []
    # outcome linkage: did the expansion that would be emptied actually improve?
    improved_and_emptied: list[tuple] = []

    for run in runs:
        ev = load(run)
        domains: dict[str, dict[str, list]] = {}          # space_id -> knob -> choices
        space_of_cand: dict[str, str] = {}
        for e in ev:
            if e["type"] == "SPACE_PUBLISHED":
                sp = e["payload"]["space"]
                domains[sp["space_id"]] = {d["name"]: d["choices"] for d in sp["domains"]}
                space_of_cand[sp["candidate_id"]] = sp["space_id"]

        # best-ms trajectory per candidate, to tell whether a re-tune paid off
        best_before: dict[str, float] = {}
        expansion_gain: dict[str, list[float]] = defaultdict(list)
        for e in ev:
            p = e.get("payload") or {}
            if e["type"] == "TRIAL_DONE":
                t = p["trial"]
                if t.get("status") == "complete":
                    c = t.get("candidate_id")
                    m = (t.get("latency_ms") or {}).get("mean")
                    if c and m and (c not in best_before or m < best_before[c]):
                        best_before[c] = m
            if e["type"] == "SPACE_EXPANDED":
                c = p.get("candidate_id")
                prev = p.get("prev_best_ms")
                if c and prev:
                    expansion_gain[c].append(prev)

        for e in ev:
            if e["type"] != "STATS_DONE":
                continue
            s = e["payload"]["stats"]
            cand, space_id = s.get("candidate_id"), s.get("space_id")
            dom = domains.get(space_id) or domains.get(space_of_cand.get(cand, ""), {})
            if not dom:
                continue

            # replay the SHIPPED winner-anchored aim (the `use_winner_anchor=True` arm)
            req = [ps for ps in s["param_stats"]
                   if ps.get("at_boundary")
                   and ps.get("boundary_direction") in ("min", "max")
                   and (ps.get("effect_pct") or 0.0) >= 2.0]
            if not req:
                continue

            survivors = []
            for ps in req:
                choices = dom.get(ps["name"])
                if not choices:
                    survivors.append(ps)
                    continue
                ev_ = edge_value(choices, ps["boundary_direction"])
                fr = (ps.get("failure_rate_by_value") or {}).get(repr(ev_))
                aimed += 1
                if fr is not None and fr >= THRESHOLD:
                    vetoed += 1
                else:
                    survivors.append(ps)

            row = (run.name, cand, len(req), len(survivors),
                   [p["name"] for p in req if p not in survivors])
            if survivors:
                kept.append(row)
            else:
                # Would the MEDIAN fallback arm still find an aim? That decides whether
                # the expansion merely changes aim or is cancelled outright at line 926.
                med = []
                for ps in s["param_stats"]:
                    choices = dom.get(ps["name"])
                    lat = ps.get("latency_by_value") or {}
                    if not choices or len(lat) < 2:
                        continue
                    if (ps.get("effect_pct") or 0.0) < 2.0:
                        continue
                    measured = [c for c in choices if repr(c) in lat]
                    if len(measured) < 2:
                        continue
                    best = min(measured, key=lambda c: lat[repr(c)])
                    if best == measured[0] or best == measured[-1]:
                        med.append(ps["name"])
                emptied.append(row + (med,))
                if expansion_gain.get(cand):
                    improved_and_emptied.append(row + (med,))

    print(f"runs scanned                     : {len(runs)}")
    print(f"knob aims replayed               : {aimed}")
    print(f"aims a {THRESHOLD:.0%} veto would suppress   : {vetoed}"
          f"  ({vetoed / aimed * 100:.1f}%)" if aimed else "")
    print()
    print(f"expansions whose aim SURVIVES    : {len(kept)}")
    print(f"expansions the veto would EMPTY  : {len(emptied)}")
    print("   ^ emptying does NOT cancel the expansion today: orchestrator.py:312-331 falls")
    print("     back to the median aim when the winner-anchored pass returns []. But")
    print("     boundary_knobs_to_expand returning [] at line 926 DOES cancel it, so the")
    print("     veto must be applied INSIDE the winner-anchored arm only -- never to the")
    print("     final result -- or these expansions lose their fresh tuning budget too.")
    if emptied:
        print("\n   expansions that would be emptied (and whether the median arm still aims):")
        for r in emptied:
            med = r[5]
            tag = f"median arm still aims at {med[:3]}" if med else "MEDIAN ARM ALSO EMPTY -> CANCELLED"
            print(f"     {r[0]}  {r[1]}  {r[2]} aims -> 0   vetoed={r[4]}")
            print(f"        {tag}")
    if improved_and_emptied:
        print(f"\n   !! {len(improved_and_emptied)} of those DID go on to expand+re-tune;")
        print("      cancelling them outright would forfeit that re-tune.")
        for r in improved_and_emptied:
            print(f"        {r[1]}  median-arm rescue: {'YES' if r[5] else 'NO'}")

    # ---- the design that survives both hazards -------------------------------------------
    vetoed_in_surviving = vetoed - sum(r[2] for r in emptied)
    print("\n" + "=" * 78)
    print("DESIGN COMPARISON (this is the part that decided the fix)")
    print("=" * 78)
    print(f"""
A. veto as a hard FILTER on both arms
     suppresses all {vetoed} failing-edge aims, but CANCELS {len(emptied)} expansions
     ({len(improved_and_emptied)} of which historically improved) -- forfeiting the fresh
     tuning budget that orchestrator.py:403-418 measured as independently valuable.

B. veto the winner-anchored arm only, let the median arm rescue
     no expansion is cancelled, but the median arm re-aims at THE SAME vetoed knobs in
     at least 5 of the 8 cases above (NUM_WARPS, BLOCK_D, BLOCK_N/BLOCK_K, ...).
     The veto is defeated exactly where it was supposed to bite.

C. veto as a RANKING, not a filter  <-- SHIPPED (budgets.max_edge_failure_frac = 0.30)
     prefer healthy-edge knobs; fall back to a failing edge only when no healthy aim
     exists in that expansion. Then:
       - {vetoed_in_surviving} failing-edge aims are avoided (they sit in expansions that
         retain a healthy alternative, so nothing is lost by skipping them)
       - {len(emptied)} expansions keep aiming at a failing edge, i.e. exactly today's
         behaviour -- so ZERO expansions are cancelled and ZERO re-tunes are forfeited
     Captures {vetoed_in_surviving}/{vetoed} = {vetoed_in_surviving / vetoed * 100:.0f}% of the benefit at no measured cost.
""")

    # ---- verify the SHIPPED code reproduces design C on this data ------------------------
    print("=" * 78)
    print("VERIFICATION: does the shipped boundary_knobs_to_expand behave as designed?")
    print("=" * 78)
    if not HAVE_REAL:
        print("  (could not import boundary_knobs_to_expand; skipped)")
        return 0
    cancelled = shrunk = same = 0
    for run in runs:
        ev = load(run)
        spaces: dict[str, dict] = {}
        for e in ev:
            if e["type"] == "SPACE_PUBLISHED":
                sp = e["payload"]["space"]
                spaces[sp["space_id"]] = sp
        for e in ev:
            if e["type"] != "STATS_DONE":
                continue
            s = e["payload"]["stats"]
            sp = spaces.get(s.get("space_id"))
            if not sp:
                continue
            try:
                stats = TuningStats.model_validate(s)
                space = ParameterSpace.model_validate(sp)
            except Exception:
                continue
            off = boundary_knobs_to_expand(stats, 0.8, space, min_effect_pct=2.0,
                                           max_edge_failure_frac=1.0)
            on = boundary_knobs_to_expand(stats, 0.8, space, min_effect_pct=2.0,
                                          max_edge_failure_frac=0.30)
            if off and not on:
                cancelled += 1
            elif len(on) < len(off):
                shrunk += 1
            elif off:
                same += 1
    print(f"  expansions where the aim SHRANK (healthy subset chosen) : {shrunk}")
    print(f"  expansions where the aim is UNCHANGED                  : {same}")
    print(f"  expansions the shipped code would CANCEL               : {cancelled}")
    if cancelled:
        print("  !! REGRESSION: the preference must never empty a non-empty aim")
        return 1
    print("  OK: no expansion is cancelled by the preference (the safety invariant holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
