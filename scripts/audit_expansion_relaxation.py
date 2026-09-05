"""Is a space expansion actually a RELAXATION of the space it replaces?

`_maybe_expand_space` documents its own assumption in a comment:

    "An expansion only ADDS choices, so the pre-expansion optimum is still a
     legal config -- but the re-tune starts a FRESH TPE study ..."

Nothing enforces that. The parameterizer returns a whole new space (it may also
rewrite the kernel body), so it is free to drop a choice, drop a constraint, or
add a constraint that excludes the very config being carried forward. This
script checks the assumption against every expansion on record, in three parts:

  1. DOMAIN SUPERSET  -- is every pre-expansion choice still offered?
  2. INCUMBENT FEASIBLE -- does the pre-expansion best params still pass the
     guard against the NEW space? (This is the check the driver already
     performs at orchestrator.py:904 and then silently discards.)
  3. SEMANTIC RELAXATION -- on the sub-grid both spaces share, does the new space
     ADMIT configurations the old one rejected? This is the real question; a raw
     constraint-text diff overstates it badly, because an expansion may re-express
     three bounds as one `and`-chain and lose nothing. Only a semantic check
     distinguishes that from a genuine drop.

  4. RESTORE SIMULATION -- replay `Orchestrator._restore_dropped_constraints` over
     the recorded pair and re-measure 3, so the effect of the fix is visible on
     historical data rather than only on new runs.

Note that a constraint drop is not automatically a defect: an expansion may
legitimately rewrite the body, and a bound derived from the old body can be wrong
for the new one (observed once -- a rewritten kernel made an N tile of 8 legal, and
the old `% 16 == 0` rule would have vetoed the only value being added). That is why
the restore is gated on each newly added choice remaining reachable.

Usage: python scripts/audit_expansion_relaxation.py [runs_dir]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import DeviceLimits  # noqa: E402
from kernel_optimizer.control.orchestrator import Orchestrator  # noqa: E402
from kernel_optimizer.models.core import ParameterSpace, ParamSet  # noqa: E402
from kernel_optimizer.paramspace.guard import check_config  # noqa: E402


def newly_admitted_frac(old: ParameterSpace, new: ParameterSpace,
                        device: DeviceLimits, rnd: random.Random,
                        n: int = 3000) -> float:
    """% of the OLD domain that NEW admits and OLD rejected (the relaxation leak)."""
    ch = {d.name: list(d.choices) for d in old.domains}
    names = list(ch)
    leaked = total = 0
    for _ in range(n):
        ps = ParamSet(values={k: rnd.choice(ch[k]) for k in names})
        was_ok = check_config(old, ps, device) is None
        try:
            now_ok = check_config(new, ps, device) is None
        except Exception:
            continue
        total += 1
        if now_ok and not was_ok:
            leaked += 1
    return 100.0 * leaked / total if total else 0.0


def load(run: Path) -> list[dict]:
    p = run / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    device = DeviceLimits()

    rows: list[dict] = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        ev = load(run)
        if not ev:
            continue
        # spaces in publication order, per candidate
        spaces: dict[str, list[dict]] = {}
        for e in ev:
            if e["type"] == "SPACE_PUBLISHED":
                sp = e["payload"]["space"]
                spaces.setdefault(sp["candidate_id"], []).append(sp)
        # best complete trial per (candidate, space)
        best: dict[tuple[str, str], dict] = {}
        for e in ev:
            if e["type"] != "TRIAL_DONE":
                continue
            t = e["payload"].get("trial") or e["payload"]
            if t.get("status") != "complete" or not t.get("latency_ms"):
                continue
            k = (t["candidate_id"], t["space_id"])
            cur = best.get(k)
            if cur is None or t["latency_ms"]["mean"] < cur["latency_ms"]["mean"]:
                best[k] = t

        for e in ev:
            if e["type"] != "SPACE_EXPANDED":
                continue
            cid = e["payload"]["candidate_id"]
            seq = spaces.get(cid, [])
            # the expansion event fires right after the new space is published,
            # so the last two published spaces for this candidate are old -> new
            new = next((s for s in reversed(seq)
                        if s["version"] == max(x["version"] for x in seq)), None)
            olds = [s for s in seq if new and s["version"] == new["version"] - 1]
            if not new or not olds:
                continue
            old = olds[-1]
            o_dom = {d["name"]: list(d["choices"]) for d in old["domains"]}
            n_dom = {d["name"]: list(d["choices"]) for d in new["domains"]}

            lost_knobs = sorted(set(o_dom) - set(n_dom))
            lost_choices = {k: sorted(set(o_dom[k]) - set(n_dom.get(k, [])), key=str)
                            for k in o_dom if set(o_dom[k]) - set(n_dom.get(k, []))}
            o_c = {c["expr"] for c in old["constraints"]}
            n_c = {c["expr"] for c in new["constraints"]}

            inc = best.get((cid, old["space_id"]))
            feasible = None
            reason = ""
            if inc is not None:
                try:
                    space = ParameterSpace.model_validate(new)
                    ps = ParamSet.model_validate(inc["params"])
                    err = check_config(space, ps, device)
                    feasible = err is None
                    reason = "" if err is None else str(err)[:110]
                except Exception as exc:  # pragma: no cover - diagnostic
                    reason = f"check raised: {exc}"[:110]

            # semantic relaxation, and what the restore fix does about it
            rnd = random.Random(f"{cid}:{old['space_id']}")
            o_space = ParameterSpace.model_validate(old)
            n_space = ParameterSpace.model_validate(new)
            leak_before = newly_admitted_frac(o_space, n_space, device, rnd)
            fixed = ParameterSpace.model_validate(new)
            stub = SimpleNamespace(cfg=SimpleNamespace(device=device))
            stub._choice_is_reachable = Orchestrator._choice_is_reachable.__get__(stub)
            restored = Orchestrator._restore_dropped_constraints(stub, o_space, fixed)
            leak_after = newly_admitted_frac(
                o_space, fixed, device, random.Random(f"{cid}:{old['space_id']}"))
            o_ch = {d["name"]: set(d["choices"]) for d in old["domains"]}
            added = [(d.name, v) for d in fixed.domains
                     for v in d.choices if v not in o_ch.get(d.name, set())]
            unreachable = [f"{k}={v}" for k, v in added
                           if not stub._choice_is_reachable(fixed, k, v)]

            rows.append({
                "run": run.name.replace("run-", "").replace("20260", "0"),
                "cand": cid,
                "leak_before": leak_before,
                "leak_after": leak_after,
                "n_restored": len(restored),
                "n_added_choices": len(added),
                "unreachable_after_restore": unreachable,
                "requested": [k["name"] for k in e["payload"].get("knobs", [])],
                "lost_knobs": lost_knobs,
                "lost_choices": lost_choices,
                "dropped_constraints": len(o_c - n_c),
                "added_constraints": len(n_c - o_c),
                "n_old_constraints": len(o_c),
                "incumbent_ms": (inc["latency_ms"]["mean"] if inc else None),
                "incumbent_feasible": feasible,
                "infeasible_reason": reason,
            })

    if not rows:
        print("no SPACE_EXPANDED events found")
        return 0

    print(f"expansions on record: {len(rows)}\n")
    hdr = f"{'run':22s} {'candidate':16s} {'lost':5s} {'drop':>4s} {'add':>4s} {'incumbent':>10s} {'feasible':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lost = "YES" if (r["lost_knobs"] or r["lost_choices"]) else "-"
        feas = {True: "yes", False: "**NO**", None: "n/a"}[r["incumbent_feasible"]]
        inc = f"{r['incumbent_ms']:.2f}" if r["incumbent_ms"] else "-"
        print(f"{r['run']:22s} {r['cand']:16s} {lost:5s} "
              f"{r['dropped_constraints']:>4d} {r['added_constraints']:>4d} {inc:>10s} {feas:>9s}")

    print()
    n_not_superset = sum(1 for r in rows if r["lost_knobs"] or r["lost_choices"])
    n_infeasible = sum(1 for r in rows if r["incumbent_feasible"] is False)
    n_dropped = sum(1 for r in rows if r["dropped_constraints"] > 0)
    print(f"INVARIANT 1  domain is a superset of the pre-expansion domain: "
          f"{len(rows) - n_not_superset}/{len(rows)} hold")
    print(f"INVARIANT 2  pre-expansion incumbent still feasible:          "
          f"{len(rows) - n_infeasible}/{len(rows)} hold")
    print(f"INFO         expansions that dropped >=1 constraint (text):    {n_dropped}/{len(rows)}")

    # --- 3 and 4: the semantic picture, and what the restore does to it -------
    leaked = [r for r in rows if r["leak_before"] > 0.05]
    still = [r for r in rows if r["leak_after"] > 0.05]
    mean_b = sum(r["leak_before"] for r in rows) / len(rows)
    mean_a = sum(r["leak_after"] for r in rows) / len(rows)
    print()
    print(f"{'run':22s} {'candidate':16s} {'leak':>7s} {'after fix':>9s} {'restored':>8s} {'new ch':>6s}")
    print("-" * 74)
    for r in rows:
        print(f"{r['run']:22s} {r['cand']:16s} {r['leak_before']:>6.1f}% "
              f"{r['leak_after']:>8.1f}% {r['n_restored']:>8d} {r['n_added_choices']:>6d}")
    print()
    print(f"SEMANTIC     expansions that ADMIT configs the old space excluded: "
          f"{len(leaked)}/{len(rows)} (mean {mean_b:.1f}% of the shared sub-grid)")
    print(f"AFTER FIX    same, with dropped constraints restored:              "
          f"{len(still)}/{len(rows)} (mean {mean_a:.1f}%)")
    print(f"             constraints restored: {sum(r['n_restored'] for r in rows)}; "
          f"newly added choices made unreachable: "
          f"{sum(len(r['unreachable_after_restore']) for r in rows)} of "
          f"{sum(r['n_added_choices'] for r in rows)}")
    for r in rows:
        if r["unreachable_after_restore"]:
            print(f"             {r['run']} {r['cand']} lost: "
                  f"{r['unreachable_after_restore']}")

    for r in rows:
        if r["lost_knobs"] or r["lost_choices"] or r["incumbent_feasible"] is False:
            print(f"\n  {r['run']} {r['cand']}")
            if r["lost_knobs"]:
                print(f"    knobs removed outright: {r['lost_knobs']}")
            for k, v in r["lost_choices"].items():
                print(f"    choices removed from {k}: {v}")
            if r["incumbent_feasible"] is False:
                print(f"    incumbent {r['incumbent_ms']:.2f} ms is now INFEASIBLE: "
                      f"{r['infeasible_reason']}")

    worst = max(rows, key=lambda r: r["dropped_constraints"])
    if worst["dropped_constraints"]:
        print(f"\nlargest constraint drop: {worst['run']} {worst['cand']} — "
              f"{worst['dropped_constraints']} of {worst['n_old_constraints']} dropped, "
              f"{worst['added_constraints']} added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
