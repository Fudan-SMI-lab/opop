"""Why did S7's live recomputes find NO wall, with four shared-memory refusals on the books?

THE QUESTION. The treatment arm recomputed at trial 10 and 20 and both times reported
`n_skipped_no_wall`, i.e. `SlopeGuide._walls` returned an empty list, with `n_skipped_no_value_toward_wall
== 0` -- so `rows` itself was empty and no candidate target was ever rejected. That has three possible
causes and they call for different responses:

  A. `find_walls` found no wall. The candidate's refusals are not attributable to a single knob (they
     need a knob whose range is TRUNCATED, not merely some refusals). Expected, declared, not a defect.
  B. `find_walls` found walls and `select_for_probing` dropped all of them -- the monotone-tail and
     positive-gain filter, measured to drop 3 of 6. Also expected, and the SHIPPING filter, which S7
     calls rather than re-implements precisely so the two cannot drift.
  C. Something in the in-loop path differs from the offline replay that priced this mechanism at 17.1%.
     That WOULD be a defect, and it is the only one of the three that is actionable mid-run.

Distinguishing them needs the two stages reported SEPARATELY over the same prefixes the live run used.
A probe that only printed the final count would read identically for all three -- the shape recorded in
`a-constant-reading-is-a-broken-probe`.

POSITIVE CONTROL, and this probe is worthless without it: the same replay runs over the CONTROL arm,
whose 2e fired at end-of-tuning and produced `RESOURCE_WALL_ATTRIBUTED`. If the replay finds no wall
there either, the replay is broken and its silence about the treatment arm means nothing.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/s7_no_wall_why.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/work/opop/src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.evaluation import soft_wall as soft_wall_mod  # noqa: E402
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.evaluation.wall_attribution import _as_num  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    LatencyStats, ParamDomain, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
)
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer  # noqa: E402

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
CFG_DIR = Path("/root/autodl-tmp/work/opop/configs")
TREAT_CFG = str(CFG_DIR / "experiments_s7_treatment_box4gpu0.yaml")
CONTROL_CFG = str(CFG_DIR / "experiments_s7_control_box4gpu1.yaml")


def _events(run: Path) -> list[dict]:
    out = []
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _spaces(ev: list[dict]) -> dict[str, ParameterSpace]:
    """EVERY declared space, keyed by `space_id` -- not just the first.

    THE DEFECT THIS FIXES, found live. The first version returned only the first `SPACE_PUBLISHED` and
    the caller sliced `trials[:cut]` over all candidates pooled, so once a run reached its SECOND
    candidate the replay was scoring candidate 2's trials against candidate 1's SPACE and candidate 1's
    per-value curves. It printed a 6-knob domain list for a 16-knob space and reported the first
    candidate's wall arithmetic verbatim at every later prefix -- three identical blocks that looked
    like a stable finding and were in fact one stale one. A wall is a statement about ONE knob of ONE
    candidate's space; pooling candidates cannot produce a true one.

    Reconstructing a space from drawn values is still refused: a value the tuner never drew is absent
    from the draws by definition, and those are exactly the values a proposal is made of.
    """
    out: dict[str, ParameterSpace] = {}
    for e in ev:
        if e.get("type") != "SPACE_PUBLISHED":
            continue
        sp = (e.get("payload") or {}).get("space") or {}
        doms = []
        for d in sp.get("domains") or []:
            doms.append(ParamDomain(name=d["name"], kind=d["kind"], choices=list(d["choices"])))
        sid = str(sp.get("space_id") or "")
        if doms and sid:
            out[sid] = ParameterSpace(
                space_id=sid, candidate_id=sp.get("candidate_id", "cand-?"),
                version=int(sp.get("version", 1)),
                source_sha=sp.get("source_sha", "0" * 64), domains=doms, constraints=[])
    return out


def _trials(ev: list[dict]) -> list[TrialRecord]:
    out: list[TrialRecord] = []
    for e in ev:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        vals = (t.get("params") or {}).get("values")
        if not vals:
            continue
        lat = t.get("latency_ms")
        prof = t.get("profile")
        out.append(TrialRecord(
            trial_id=t.get("trial_id", "tr-?"), candidate_id=t.get("candidate_id", "cand-?"),
            space_id=t.get("space_id", "sp-?"), params=ParamSet(values=dict(vals)),
            status=t.get("status", "fail"), failure_kind=t.get("failure_kind"),
            latency_ms=LatencyStats(**{k: lat[k] for k in
                                       ("mean", "median", "min", "max", "std", "n_samples")
                                       if k in lat}) if isinstance(lat, dict) else None,
            profile=ProfileRecord(**{k: v for k, v in prof.items()
                                     if k in ProfileRecord.model_fields})
            if isinstance(prof, dict) else None,
        ))
    return out


def report(arm: str, cfg_path: str, every: int = 10) -> None:
    """Replay each space SEPARATELY, at that space's own recompute points.

    Per space, not per run: S7 builds a fresh `SlopeGuide` for every space and its cadence counts that
    space's own `n_told`, so a replay that pools candidates asks the question at the wrong places with
    the wrong curves. `every` mirrors `recompute_every`, and the prefixes are derived from each space's
    trial count rather than hardcoded, so a space that closed early is not replayed past its end.
    """
    runs = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    if not runs:
        print(f"=== {arm}: no run")
        return
    run = runs[-1]
    ev = _events(run)
    spaces = _spaces(ev)
    trials = _trials(ev)
    print(f"=== {arm}  {run.name}   spaces declared {len(spaces)}   trials {len(trials)}")
    if not spaces:
        return

    # The ARM'S OWN device limits, loaded from the config the arm ran with. `max_shared_bytes_optin`
    # is what makes a refusal a refusal (4090: 101376), so a guessed limit would invent walls or erase
    # them -- and this replay's whole purpose is to say whether a wall exists.
    cfg = load_config(Path(cfg_path))
    print(f"    device: shared_optin={cfg.device.max_shared_bytes_optin} "
          f"regs={cfg.device.max_regs_per_thread}")
    analyzer = TuningStatsAnalyzer(cfg.device)

    for sid, space in spaces.items():
        own = [t for t in trials if t.space_id == sid]
        print(f"  -- space {sid}  cand={space.candidate_id} v{space.version}  "
              f"{len(space.domains)} knobs  {len(own)} trials")
        print("     domains: " + ", ".join(f"{d.name}={d.choices}" for d in space.domains))
        if not own:
            print("     (no trials yet)")
            continue
        cuts = [c for c in range(every, len(own) + 1, every)] or [len(own)]
        if cuts[-1] != len(own):
            cuts.append(len(own))
        _one_space(analyzer, space, own, cuts)


def _one_space(analyzer: TuningStatsAnalyzer, space: ParameterSpace,
               trials: list[TrialRecord], cuts: list[int]) -> None:
    for cut in cuts:
        sub = trials[:cut]
        refused = [t.params.values for t in sub
                   if t.failure_kind == "infeasible_shared_memory" and t.params]
        stats = analyzer.analyze(space, sub)
        walls = wall_attribution.find_walls(stats, refused) if refused else []
        worth, _ = wall_attribution.select_for_probing(walls, -1) if walls else ([], [])
        soft = soft_wall_mod.find_soft_walls(stats, sub)

        print(f"    -- prefix {cut}: refusals {len(refused)}   "
              f"find_walls -> {len(walls)}   select_for_probing keeps {len(worth)} "
              f"soft walls {len(soft.walls)}")
        # NOT printing `len(dropped)`: with max_probes=-1 `select_for_probing` returns `[]` as the
        # skipped list BY CONSTRUCTION, so "drops 0" is a constant that reads as information -- the
        # `a-constant-reading-is-a-broken-probe` shape. The verdict is in `monotone`/`tail_gain_pct`,
        # printed per wall below, which is what the filter actually tests.
        for w in walls:
            kept = any(k.param == w.param and k.side == w.side for k in worth)
            print(f"         HARD {w.param} side={w.side} refused={w.refused_value} "
                  f"gain={w.tail_gain_pct:.1f}% monotone={w.monotone} "
                  f"tail={w.tail_values}->{[round(x, 2) for x in w.tail_latencies]}  "
                  f"{'KEPT' if kept else 'DROPPED by the slope filter'}")
        for sw in soft.walls:
            print(f"         SOFT {sw.param} gain={sw.tail_gain_pct:.1f}%")
        if not refused:
            print("         (no refusals in this prefix -- find_walls has no input at all)")
        elif not walls:
            # Report the reason rather than only the count. A refused value stops being a truncation
            # the moment the tuner MEASURES it beside a different partner, so walls are not
            # cumulative -- measured live: NUM_WARPS=16 refused at trial 6, completed at trial 14,
            # wall gone from prefix 20 on. Printing the per-knob arithmetic makes that visible
            # instead of leaving "0 walls" to be read as a defect.
            print("         => cause A: refusals exist, none is OUTSIDE its knob's measured range")
            for ps in stats.param_stats:
                ran = sorted(v for k, v in ((k, _as_num(k)) for k in
                                            (ps.latency_by_value or {})) if v is not None)
                refs = sorted({v for p in refused if (v := _as_num(p.get(ps.name))) is not None})
                if not refs or not ran:
                    continue
                out = [r for r in refs if r > max(ran) or r < min(ran)]
                print(f"            {ps.name}: measured {ran}  refused {refs}  outside {out}")
        elif not worth and not soft.walls:
            print("         => cause B: walls found, the shipping slope filter dropped every one "
                  "(latency flat or RISING toward the wall, so freeing it buys nothing)")


def main() -> int:
    # The live recompute points, so the replay asks the question at the same places the run did.
    report("s7-treatment", TREAT_CFG)
    print()
    # POSITIVE CONTROL: this arm's 2e fired and emitted RESOURCE_WALL_ATTRIBUTED. A replay that finds
    # nothing here is a broken replay, and would void every negative above.
    report("s7-control", CONTROL_CFG)
    print()
    print("READ THE CONTROL ARM FIRST. If it shows no wall, this probe proves nothing about the")
    print("treatment arm -- the run's own events say a wall WAS attributed there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
