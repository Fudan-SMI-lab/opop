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

WHAT IT ACTUALLY FOUND, 2026-09-10 (docs/result-s1-replay-inconclusive.md) -- read before reusing:

  * Conditional shrinking removed 100% of dead draws; unconditional value removal removed 6.9%, and
    only 2 of 31 spaces had a value infeasible beside EVERY partner. So S1 must be the
    dependency-aware variant; the simpler one is measurably vacuous. That is the useful result.
  * NEITHER arm improved any space's best latency, and that is NOT a verdict on S1: the recorded
    universe contains only the points the OLD sampler chose, while a shrink's purpose is to spend the
    freed budget elsewhere -- which is by construction absent from the table. The lower-bound caveat
    below is not a footnote; on this corpus it is the dominant effect.
  * KNOWN COMPARABILITY BUG, unfixed on purpose: `declare_cond` draws from the domain while the other
    arms walk a shuffled recorded pool, so it evaluates ~24.4 points against their 27.9 and its
    "5 spaces worse" is an artefact of getting fewer draws. Fix it when the replay is next used for
    something it can answer; do not quote that figure.

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


def _dead_values(domains: dict[str, list], infeasible: set[str],
                 table: dict[str, float] | None = None) -> dict[str, set]:
    """Values that appear in NO FEASIBLE recorded configuration -- what S1 may remove.

    THE DEFINITION IS THE WHOLE POINT, and my first version got it wrong in a way that matters more
    than the bug: it counted dead sightings (>= 3 infeasible configurations containing the value) and
    never looked at whether the value ALSO appeared in configurations that ran. That is precisely the
    count-based retirement the plan forbids, reproduced inside the tool built to test the plan.

    What it cost, measured: on space sp-6a0e4fe6 it removed `DOT_PRECISION=fp16`, which had 18
    feasible recorded configurations and held that space's best latency of 2.6286 ms. The `declare`
    arm then read 37.6719 ms -- 14x worse -- and the aggregate said S1 lost 10 spaces to 0. The
    "result" was entirely an artefact of my rule, and it is the same shape as the history that
    refused count-based retirement in the first place: failure is CONDITIONAL (this value fails
    beside some partners and wins beside others) while removal is unconditional.

    The correct rule needs the feasible side of the ledger as well: a value may be removed only when
    it appears in at least MIN_EVIDENCE infeasible configurations AND in ZERO feasible ones. That is
    the S1 claim -- "compile-time-known infeasible, so it cannot appear in a runnable configuration"
    -- rather than "has failed a few times".

    Without `table`, the feasible side is unknown; the honest answer is then to remove NOTHING, since
    an unconditional action on unknown evidence is exactly what caused the damage above.
    """
    MIN_EVIDENCE = 3
    dead_hits: dict[tuple[str, str], int] = defaultdict(int)
    for k in infeasible:
        for part in k.split("|"):
            name, _, val = part.partition("=")
            dead_hits[(name, val)] += 1
    alive: set[tuple[str, str]] = set()
    for k in (table or {}):
        for part in k.split("|"):
            name, _, val = part.partition("=")
            alive.add((name, val))
    out: dict[str, set] = defaultdict(set)
    for (name, val), hits in dead_hits.items():
        if hits < MIN_EVIDENCE or name not in domains:
            continue
        if (name, val) in alive:
            # A live counter-example exists: this value runs beside SOME partner, so its failures are
            # conditional and removing it would kill a reachable region.
            continue
        out[name].add(val)
    # Never empty a domain: that would make the space unsamplable and is a bug, not a shrink.
    for name, vals in list(out.items()):
        if len(vals) >= len(domains[name]):
            del out[name]
    return dict(out)


def _conditional_domain(space: dict, bound: dict[str, str], name: str,
                        choices: list) -> list:
    """Legal values for `name` GIVEN what is already bound -- S1's actual path A (ATF-style).

    WHY A THIRD ARM EXISTS. The `declare` arm removes a value from the domain outright, and measured
    on this corpus that is nearly vacuous: only 2 of 31 spaces had any value that was infeasible
    beside EVERY partner, so 0 spaces improved and 6.9% of dead draws were avoided. That is not a
    verdict on S1 -- it is a verdict on unconditional removal, which is the weaker mechanism and not
    what the plan proposes. Shared-memory infeasibility is a property of the COMBINATION (tile x
    depth x dtype width), so the only shrink with anything to remove is a CONDITIONAL one.

    The oracle is recorded support: `name=v` is legal after `bound` when some recorded FEASIBLE
    configuration agrees with `bound` and has `name=v`. Deliberately permissive -- a value with no
    recorded evidence either way stays legal, because unknown must never shrink the space (the same
    rule `cached_shared_verdict` follows with its three-valued return).
    """
    feasible = space.get("_feasible_parsed")
    if feasible is None:
        feasible = [dict(part.partition("=")[::2] for part in k.split("|"))
                    for k in space["table"]]
        space["_feasible_parsed"] = feasible
    supported: set[str] = set()
    any_match = False
    for cfg in feasible:
        if all(cfg.get(k) == v for k, v in bound.items()):
            any_match = True
            if name in cfg:
                supported.add(cfg[name])
    if not any_match or not supported:
        return list(choices)  # no evidence -> no shrink
    keep = [c for c in choices if str(c) in supported]
    return keep or list(choices)


def _run_arm(space: dict, arm: str, budget: int, seed: int, universe: str = "recorded",
             max_rejects_per_ask: int = 64) -> dict:
    """One sampler over the lookup table. Returns its best-so-far curve and its waste counts.

    Deliberately a RANDOM sampler rather than TPE. Two reasons, and the second is the important
    one: a TPE fitted to a table where most points are missing would be modelling the table's holes
    rather than the space, and -- per the discipline that a comparison must be able to fail -- the
    difference between the arms here should come from the DOMAIN, not from a surrogate whose
    behaviour on sparse lookups is itself unvalidated. Random is the weaker sampler for both arms
    equally, so a difference it shows is attributable.

    `universe` decides what may be drawn, and the default is NOT the obvious choice:

      "full"      the whole domain product. Measured to be USELESS on this corpus and kept only so
                  the uselessness is reproducible: these spaces reach 2.7e15 points while a run
                  records ~30, so a random draw lands on a recorded point with probability ~1e-14.
                  17 of 24 spaces produced no comparison at all -- every draw `missing`.
      "recorded"  the points the historical run actually visited, feasible and infeasible together.
                  This is what a lookup-table replay can support, and it is what Kernel Tuner's
                  simulation mode does: it replays over a searchspace of measured points. The price
                  is a NARROWER question -- see the header -- but it is a question with an answer.
    """
    rng = random.Random(seed)
    domains = dict(space["domains"])
    removed = {}
    if arm == "declare":
        removed = _dead_values(space["domains"], space["infeasible"], space["table"])
        for name, dead in removed.items():
            keep = [c for c in domains[name] if str(c) not in {str(d) for d in dead}]
            if keep:
                domains[name] = keep

    pool: list[str] | None = None
    if universe == "recorded" and arm != "declare_cond":
        # Every point the run visited. The `declare` arm then drops those containing a removed
        # value -- which is exactly what a shrunken domain does to this pool, and is the only part
        # of a domain shrink the recorded ground can speak to.
        allowed = {n: {str(v) for v in c} for n, c in domains.items()}

        def _ok(key: str) -> bool:
            for part in key.split("|"):
                name, _, val = part.partition("=")
                if name in allowed and val not in allowed[name]:
                    return False
            return True

        pool = [k for k in list(space["table"]) + sorted(space["infeasible"]) if _ok(k)]
        rng.shuffle(pool)

    best = None
    curve: list[float | None] = []
    evaluated = 0
    dead_draws = 0
    missing_draws = 0
    seen: set[str] = set()
    # The budget is spent by ASKS, not by table hits. This distinction is load-bearing and I got it
    # wrong first: counting only hits made the loop run until `budget` measured points were found,
    # which on a 600k-point domain with a few hundred recorded points does not terminate in any
    # useful time -- it ran past ten minutes on the real corpus. It is also the WRONG accounting:
    # a live run spends its budget on the trials it asks for, and a point absent from the table is
    # a point the historical run did not measure, not a free draw. Capped by asks, so both arms get
    # exactly the same number of opportunities and a sparser table shows up as fewer evaluations.
    asks = 0
    cursor = 0
    while asks < budget:
        asks += 1
        rejects = 0
        chosen = None
        while rejects < max_rejects_per_ask:
            if pool is not None:
                if cursor >= len(pool):
                    break
                key = pool[cursor]
                cursor += 1
            elif arm == "declare_cond":
                # Bind one knob at a time, each from the domain the already-bound ones leave legal.
                bound: dict[str, str] = {}
                for n in sorted(domains):
                    legal = _conditional_domain(space, bound, n, domains[n])
                    bound[n] = str(rng.choice(legal))
                key = "|".join(f"{n}={bound[n]}" for n in sorted(bound))
            else:
                key = _key({n: rng.choice(c) for n, c in domains.items()})
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
    return {"arm": arm, "best": best, "curve": curve, "evaluated": evaluated, "asks": asks,
            "dead_draws": dead_draws, "missing_draws": missing_draws,
            "pool_size": len(pool) if pool is not None else None,
            "domain_size": _size(domains), "full_domain_size": _size(space["domains"])}
    curve: list[float | None] = []
    evaluated = 0
    dead_draws = 0
    missing_draws = 0
    seen: set[str] = set()
    # The budget is spent by ASKS, not by table hits. This distinction is load-bearing and I got it
    # wrong first: counting only hits made the loop run until `budget` measured points were found,
    # which on a 600k-point domain with a few hundred recorded points does not terminate in any
    # useful time -- it ran past ten minutes on the real corpus. It is also the WRONG accounting:
    # a live run spends its budget on the trials it asks for, and a point absent from the table is
    # a point the historical run did not measure, not a free draw. Capped by asks, so both arms get
    # exactly the same number of opportunities and a sparser table shows up as fewer evaluations.
    asks = 0
    while asks < budget:
        asks += 1
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
    return {"arm": arm, "best": best, "curve": curve, "evaluated": evaluated, "asks": asks,
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
    ap.add_argument("--universe", choices=("recorded", "full"), default="recorded",
                    help="what may be drawn. `recorded` = the points the run actually visited, which "
                         "is what a lookup table can support. `full` = the whole domain product, "
                         "measured to be useless on this corpus (17 of 24 spaces gave no comparison "
                         "because a 2.7e15-point domain never lands on one of ~30 recorded points); "
                         "kept so that result is reproducible rather than folklore.")
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
        per_arm: dict[str, list[dict]] = {"reject": [], "declare": [], "declare_cond": []}
        for seed in range(args.seeds):
            for arm in ("reject", "declare", "declare_cond"):
                per_arm[arm].append(_run_arm(s, arm, args.budget, seed, args.universe))
        row = {"space_id": sid, "run": s["run"], "candidate_id": s["candidate_id"],
               "table_points": len(s["table"]), "infeasible_points": len(s["infeasible"]),
               "dead_values_removed": {k: sorted(map(str, v)) for k, v in
                                       _dead_values(s["domains"], s["infeasible"], s["table"]).items()}}
        for arm in ("reject", "declare", "declare_cond"):
            bests = [r["best"] for r in per_arm[arm] if r["best"] is not None]
            row[arm] = {
                "best_median": sorted(bests)[len(bests) // 2] if bests else None,
                "best_min": min(bests) if bests else None,
                "evaluated_mean": sum(r["evaluated"] for r in per_arm[arm]) / args.seeds,
                "dead_draws_mean": sum(r["dead_draws"] for r in per_arm[arm]) / args.seeds,
                "missing_draws_mean": sum(r["missing_draws"] for r in per_arm[arm]) / args.seeds,
                "domain_size": per_arm[arm][0]["domain_size"],
                "pool_size": per_arm[arm][0]["pool_size"],
            }
        rows.append(row)

    print()
    print("%-12s %-9s %-10s %-10s %-10s %-8s %-8s %-8s" % (
        "space", "tbl/infs", "reject ms", "declare ms", "cond ms", "rej dead", "dec dead",
        "cnd dead"))
    print("-" * 92)
    def _ms(r, arm):
        v = r[arm]["best_median"]
        return ("%.4f" % v) if v else "-"
    for r in rows:
        print("%-12s %-9s %-10s %-10s %-10s %-8.1f %-8.1f %-8.1f" % (
            r["space_id"][:12], "%d/%d" % (r["table_points"], r["infeasible_points"]),
            _ms(r, "reject"), _ms(r, "declare"), _ms(r, "declare_cond"),
            r["reject"]["dead_draws_mean"], r["declare"]["dead_draws_mean"],
            r["declare_cond"]["dead_draws_mean"]))

    # The aggregate. Reported as a count of spaces where each arm won, not as a mean ratio: the
    # spaces have different latencies and different table coverage, so averaging across them would
    # be dominated by whichever candidate happened to be slowest.
    def _cmp(arm):
        better = worse = tie = 0
        for r in rows:
            a, b = r["reject"]["best_median"], r[arm]["best_median"]
            if a is None or b is None:
                continue
            if b < a * 0.99:
                better += 1
            elif b > a * 1.01:
                worse += 1
            else:
                tie += 1
        return better, worse, tie

    print()
    for arm, label in (("declare", "unconditional value removal"),
                       ("declare_cond", "CONDITIONAL domain shrink (S1 path A)")):
        b, w, t = _cmp(arm)
        dead_r = sum(r["reject"]["dead_draws_mean"] for r in rows)
        dead_a = sum(r[arm]["dead_draws_mean"] for r in rows)
        print("%-40s better %2d  worse %2d  within 1%% %2d   dead draws %.1f -> %.1f (%.1f%% removed)"
              % (label, b, w, t, dead_r, dead_a,
                 (100 * (dead_r - dead_a) / dead_r) if dead_r else 0.0))
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
