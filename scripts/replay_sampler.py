"""Does declaring infeasibility OUT of the space beat sampling a dead point and rejecting it? (S1)

ZERO GPU. Answered by replaying historical trials as a lookup table, in the spirit of Kernel Tuner's
simulation mode: every (configuration -> measured latency) pair a real run produced is read out of
events.jsonl, and two samplers are then run over that table under the same trial budget.

  ARM `reject`  the status quo. The sampler draws from the FULL domain product; a draw the
                compile-time screen already knows cannot launch is reported PRUNED and re-drawn, and
                the re-draw counts against a per-ask reject budget exactly as `OptunaTPETuner` does.
  ARM `declare` S1. Values that cannot appear in ANY feasible configuration are removed from the
                domain before sampling, so those points do not exist to be drawn.

WHAT THIS CAN ANSWER: under an identical trial budget, which arm's best-so-far curve is better, and
how many draws each one burns on points that could never run.

WHAT THIS CANNOT ANSWER, and it must be said in the output rather than buried here: a shrunken domain
makes TPE sample points the historical run never measured, and those are NOT in the table. So the
replay is a LOWER BOUND on the difference, not an estimate of it -- it can show that S1 does not help
on ground already covered, and it cannot show the full size of a gain. It does not replace the
control run.

WHY A LOWER BOUND IS STILL WORTH HAVING: it is free, and it is the only thing that can refuse S1
before a 12-hour pair of runs is spent on it. If `declare` cannot beat `reject` on the historical
ground, the mechanism is not doing what its motivating measurement said it would.

THE HONEST WEAKNESS OF THE FEASIBILITY MODEL. What is knowable at compile time here comes from what
the real run RECORDED as infeasible -- `CONFIG_SCREENED_INFEASIBLE` events plus trials that failed
with `infeasible_shared_memory`. That is a subset of true infeasibility (the screen only sampled part
of the space), so the replay UNDERSTATES how much S1 has to work with. Same direction as the caveat
above: this script is built to be pessimistic about S1, so that a positive result means something.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_space_trials(run_dir: Path) -> dict[str, dict]:
    """Per space: its domains, the measured latency of every configuration, and the dead points.

    Keyed by space_id because a run holds many, and a sampler comparison is only meaningful inside
    one space -- different candidates have different knobs.
    """
    spaces: dict[str, dict] = {}
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        return spaces
    for line in ev.open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, p = e.get("type"), e.get("payload") or {}
        if t == "SPACE_PUBLISHED":
            space = p.get("space") or {}
            sid = space.get("space_id")
            doms = space.get("domains") or []
            if not sid or not doms:
                continue
            spaces.setdefault(sid, {"domains": {}, "table": {}, "infeasible": set(),
                                    "run": run_dir.name,
                                    "candidate_id": space.get("candidate_id", "")})
            spaces[sid]["domains"] = {d["name"]: list(d["choices"]) for d in doms}
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or p
            sid = tr.get("space_id")
            if not sid:
                continue
            s = spaces.setdefault(sid, {"domains": {}, "table": {}, "infeasible": set(),
                                       "run": run_dir.name, "candidate_id": ""})
            vals = (tr.get("params") or {}).get("values") or {}
            if not vals:
                continue
            key = _key(vals)
            if tr.get("status") == "complete":
                lat = tr.get("latency_ms") or {}
                # The same objective the tuner optimises: median, falling back to the mean. Using
                # the mean here would make the replay disagree with the live sampler for a reason
                # that has nothing to do with the arms -- at n=20 the mean picks the faster of two
                # configurations 64.8% of the time against the median's 93.2%.
                ms = lat.get("median")
                if not isinstance(ms, (int, float)):
                    ms = lat.get("mean")
                if isinstance(ms, (int, float)) and ms > 0:
                    s["table"][key] = float(ms)
            elif tr.get("failure_kind") == "infeasible_shared_memory":
                s["infeasible"].add(key)
        elif t == "CONFIG_SCREENED_INFEASIBLE":
            # These carry no space_id, so they are attributed by candidate. Recorded against every
            # space of that candidate: a screened point is a property of the SOURCE, and attributing
            # it too widely can only make `reject` look better than it is, which is the safe
            # direction for this comparison.
            vals = p.get("params") or {}
            if vals:
                for s in spaces.values():
                    if s.get("candidate_id") and s["candidate_id"] == p.get("candidate_id"):
                        s["infeasible"].add(_key(vals))
    return spaces


def _key(values: dict) -> str:
    return "|".join(f"{k}={values[k]}" for k in sorted(values))


def _dead_values(domains: dict[str, list], infeasible: set[str]) -> dict[str, set]:
    """Values that appear in NO feasible recorded configuration -- what S1 could remove.

    This is the conservative reading of "declare it out of the space". A value is only removed when
    every recorded configuration containing it was infeasible AND at least `MIN_EVIDENCE` such
    configurations were seen; a value with one dead sighting proves nothing, and removing it would
    be the unconditional-retirement mistake that measured 38.5% wrongful kills on real history.
    """
    MIN_EVIDENCE = 3
    dead_hits: dict[tuple[str, object], int] = defaultdict(int)
    for k in infeasible:
        for part in k.split("|"):
            name, _, val = part.partition("=")
            dead_hits[(name, val)] += 1
    out: dict[str, set] = defaultdict(set)
    for (name, val), hits in dead_hits.items():
        if hits >= MIN_EVIDENCE and name in domains:
            out[name].add(val)
    # Never empty a domain: that would make the space unsamplable and is a bug, not a shrink.
    for name, vals in list(out.items()):
        if len(vals) >= len(domains[name]):
            del out[name]
    return dict(out)


def _run_arm(space: dict, arm: str, budget: int, seed: int,
             max_rejects_per_ask: int = 64) -> dict:
    """One sampler over the lookup table. Returns its best-so-far curve and its waste counts.

    Deliberately a RANDOM sampler rather than TPE. Two reasons, and the second is the important
    one: a TPE fitted to a table where most points are missing would be modelling the table's holes
    rather than the space, and -- per the discipline that a comparison must be able to fail -- the
    difference between the arms here should come from the DOMAIN, not from a surrogate whose
    behaviour on sparse lookups is itself unvalidated. Random is the weaker sampler for both arms
    equally, so a difference it shows is attributable.
    """
    rng = random.Random(seed)
    domains = dict(space["domains"])
    if arm == "declare":
        for name, dead in _dead_values(space["domains"], space["infeasible"]).items():
            keep = [c for c in domains[name] if str(c) not in {str(d) for d in dead}]
            if keep:
                domains[name] = keep

    best = None
    curve: list[float | None] = []
    evaluated = 0
    dead_draws = 0
    missing_draws = 0
    seen: set[str] = set()
    while evaluated < budget:
        rejects = 0
        chosen = None
        while rejects < max_rejects_per_ask:
            vals = {n: rng.choice(c) for n, c in domains.items()}
            key = _key(vals)
            if key in seen:
                rejects += 1
                continue
            if key in space["infeasible"]:
                # The status quo: drawn, refused by the screen, re-drawn. Costs an ask, not a trial.
                dead_draws += 1
                rejects += 1
                seen.add(key)
                continue
            chosen = key
            break
        if chosen is None:
            break
        seen.add(chosen)
        ms = space["table"].get(chosen)
        if ms is None:
            # Not in the table: the historical run never measured this point. Counted, NOT
            # imputed -- inventing a latency here is exactly how a replay starts producing
            # conclusions about a space it has no data for.
            missing_draws += 1
            continue
        evaluated += 1
        if best is None or ms < best:
            best = ms
        curve.append(best)
    return {"arm": arm, "best": best, "curve": curve, "evaluated": evaluated,
            "dead_draws": dead_draws, "missing_draws": missing_draws,
            "domain_size": _size(domains), "full_domain_size": _size(space["domains"])}


def _size(domains: dict[str, list]) -> int:
    n = 1
    for c in domains.values():
        n *= max(1, len(c))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories holding events.jsonl")
    ap.add_argument("--budget", type=int, default=40, help="trials per space (trials_per_space)")
    ap.add_argument("--seeds", type=int, default=20, help="repeats per arm; the spread matters")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spaces: dict[str, dict] = {}
    for r in args.runs:
        spaces.update(_load_space_trials(Path(r)))

    usable = {sid: s for sid, s in spaces.items()
              if s["domains"] and len(s["table"]) >= 10 and s["infeasible"]}
    print("spaces found       : %d" % len(spaces))
    print("with a table and >=1 recorded infeasible point: %d" % len(usable))
    if not usable:
        print()
        print("NOTHING TO COMPARE. Either no space recorded an infeasible configuration, or no")
        print("space has enough measured points. That is a result about the CORPUS, not about S1:")
        print("with no dead points in the record there is nothing for a domain shrink to remove.")
        return 1

    rows = []
    for sid, s in sorted(usable.items()):
        per_arm: dict[str, list[dict]] = {"reject": [], "declare": []}
        for seed in range(args.seeds):
            for arm in ("reject", "declare"):
                per_arm[arm].append(_run_arm(s, arm, args.budget, seed))
        row = {"space_id": sid, "run": s["run"], "candidate_id": s["candidate_id"],
               "table_points": len(s["table"]), "infeasible_points": len(s["infeasible"]),
               "dead_values_removed": {k: sorted(map(str, v)) for k, v in
                                       _dead_values(s["domains"], s["infeasible"]).items()}}
        for arm in ("reject", "declare"):
            bests = [r["best"] for r in per_arm[arm] if r["best"] is not None]
            row[arm] = {
                "best_median": sorted(bests)[len(bests) // 2] if bests else None,
                "best_min": min(bests) if bests else None,
                "evaluated_mean": sum(r["evaluated"] for r in per_arm[arm]) / args.seeds,
                "dead_draws_mean": sum(r["dead_draws"] for r in per_arm[arm]) / args.seeds,
                "missing_draws_mean": sum(r["missing_draws"] for r in per_arm[arm]) / args.seeds,
                "domain_size": per_arm[arm][0]["domain_size"],
            }
        rows.append(row)

    print()
    print("%-12s %-9s %-6s %-10s %-10s %-9s %-9s %s" % (
        "space", "tbl/infs", "shrink", "reject ms", "declare ms", "rej dead", "dec dead", "removed"))
    print("-" * 108)
    for r in rows:
        shrink = ("%d->%d" % (r["reject"]["domain_size"], r["declare"]["domain_size"]))
        print("%-12s %-9s %-6s %-10s %-10s %-9.1f %-9.1f %s" % (
            r["space_id"][:12], "%d/%d" % (r["table_points"], r["infeasible_points"]), shrink,
            ("%.4f" % r["reject"]["best_median"]) if r["reject"]["best_median"] else "-",
            ("%.4f" % r["declare"]["best_median"]) if r["declare"]["best_median"] else "-",
            r["reject"]["dead_draws_mean"], r["declare"]["dead_draws_mean"],
            ",".join("%s:%s" % (k, "/".join(v)) for k, v in
                     r["dead_values_removed"].items())[:34] or "-"))

    # The aggregate. Reported as a count of spaces where each arm won, not as a mean ratio: the
    # spaces have different latencies and different table coverage, so averaging across them would
    # be dominated by whichever candidate happened to be slowest.
    better = worse = tie = 0
    for r in rows:
        a, b = r["reject"]["best_median"], r["declare"]["best_median"]
        if a is None or b is None:
            continue
        if b < a * 0.99:
            better += 1
        elif b > a * 1.01:
            worse += 1
        else:
            tie += 1
    dead_r = sum(r["reject"]["dead_draws_mean"] for r in rows)
    dead_d = sum(r["declare"]["dead_draws_mean"] for r in rows)
    print()
    print("SPACES WHERE `declare` BEAT `reject` BY >1%%: %d ; worse: %d ; within 1%%: %d"
          % (better, worse, tie))
    print("DEAD DRAWS (a point the screen already refused), summed over spaces:")
    print("  reject  %.1f      declare %.1f      removed %.1f%%"
          % (dead_r, dead_d, (100 * (dead_r - dead_d) / dead_r) if dead_r else 0.0))
    print()
    print("READ THIS BEFORE QUOTING THE NUMBERS ABOVE")
    print("  This is a LOWER BOUND on S1's effect, not an estimate of it. A shrunken domain makes")
    print("  the sampler visit points the historical run never measured; those are absent from the")
    print("  table and are counted as `missing`, never imputed. So a gain here is real and a")
    print("  no-difference here does NOT clear S1 -- only the control run can.")
    print("  The feasibility model is also a SUBSET of true infeasibility (it is what the run")
    print("  happened to record), which understates what S1 has to work with. Both biases point")
    print("  the same way: against S1.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print("\nwritten: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
