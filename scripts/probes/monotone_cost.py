"""How many walls does the `monotone` requirement reject, and how much slope goes with them?

WHY THIS EXISTS. The live S7 pair keeps reporting "no wall", and the corrected per-space replay showed
that is not the whole story: on the control arm's v2 expansion a `BLOCK_N` wall with a 51.1% tail gain was
found and then DROPPED, because its tail [32,64,128] -> [9.62, 40.15, 4.71] ms is not monotone. One
observation is not a rate, so this measures the whole box4 corpus.

WHAT THE FILTER IS FOR, stated first so this is not read as an argument against it. `select_for_probing`
keeps `monotone and tail_gain_pct > 0`. A wall with a FLAT or WORSENING approach is real and worthless:
freeing it buys nothing, and steering a rewrite at it spends a rewrite on a non-problem. Measured on
box 2 it removed 3 of 6 walls, the worst at -54.8%. That is a filter doing its job.

THE QUESTION IS NARROWER: of the walls it rejects, how many are rejected by `monotone` ALONE while
carrying a large positive gain? Those are the ones where a single non-monotone step -- one scheduling
spike in the middle of an otherwise descending tail -- discards a steep slope. `gain` is computed from
the tail's ENDPOINTS (`(lats[0]-lats[-1])/lats[0]`), so a wall can have a large positive gain and a
non-monotone interior at the same time; that combination is exactly what this counts.

THIS PROBE PROPOSES NOTHING AND CHANGES NOTHING. `monotone` is the shipping predicate shared by 2e, the
report, the prompt and S7. Touching it mid-pair would alter BOTH arms' old path and inject a second
independent variable. The output is a number for the next round to argue with.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/monotone_cost.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/work/opop/src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    LatencyStats, ParamDomain, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
)
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer  # noqa: E402

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
CFG = Path("/root/autodl-tmp/work/opop/configs/experiments_s7_control_box4gpu1.yaml")

# A gain this large is what makes a rejection interesting: below it, dropping the wall costs little
# whatever the reason. Stated as a constant so the count can be re-read at another threshold.
BIG_GAIN_PCT = 10.0


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
    out: dict[str, ParameterSpace] = {}
    for e in ev:
        if e.get("type") != "SPACE_PUBLISHED":
            continue
        sp = (e.get("payload") or {}).get("space") or {}
        doms = [ParamDomain(name=d["name"], kind=d["kind"], choices=list(d["choices"]))
                for d in (sp.get("domains") or [])]
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
        lat, prof = t.get("latency_ms"), t.get("profile")
        out.append(TrialRecord(
            trial_id=t.get("trial_id", "tr-?"), candidate_id=t.get("candidate_id", "cand-?"),
            space_id=t.get("space_id", "sp-?"), params=ParamSet(values=dict(vals)),
            status=t.get("status", "fail"), failure_kind=t.get("failure_kind"),
            latency_ms=LatencyStats(**{k: lat[k] for k in
                                       ("mean", "median", "min", "max", "std", "n_samples")
                                       if k in lat}) if isinstance(lat, dict) else None,
            profile=ProfileRecord(**{k: v for k, v in prof.items()
                                     if k in ProfileRecord.model_fields})
            if isinstance(prof, dict) else None))
    return out


def main() -> int:
    cfg = load_config(CFG)
    analyzer = TuningStatsAnalyzer(cfg.device)

    kept: list[float] = []
    drop_nonmono: list[tuple[float, str, str, list[float]]] = []
    drop_gain: list[float] = []
    n_spaces = n_with_refusals = 0

    for run in sorted(BASE.glob("*/run-l3-43-*")):
        ev = _events(run)
        spaces, trials = _spaces(ev), _trials(ev)
        for sid, space in spaces.items():
            own = [t for t in trials if t.space_id == sid]
            if not own:
                continue
            n_spaces += 1
            refused = [t.params.values for t in own
                       if t.failure_kind == "infeasible_shared_memory" and t.params]
            if not refused:
                continue
            n_with_refusals += 1
            stats = analyzer.analyze(space, own)
            for w in wall_attribution.find_walls(stats, refused):
                if w.monotone and w.tail_gain_pct > 0.0:
                    kept.append(w.tail_gain_pct)
                elif not w.monotone and w.tail_gain_pct > 0.0:
                    drop_nonmono.append((w.tail_gain_pct, w.param,
                                         f"{run.parent.name}/{sid}",
                                         [round(x, 2) for x in w.tail_latencies]))
                else:
                    drop_gain.append(w.tail_gain_pct)

    total = len(kept) + len(drop_nonmono) + len(drop_gain)
    print(f"spaces replayed {n_spaces}, of which {n_with_refusals} had shared-memory refusals")
    print(f"walls found across the corpus: {total}")
    if not total:
        print("no walls at all -- nothing to say about the filter")
        return 0
    print(f"  KEPT (monotone and gain > 0):        {len(kept)}  "
          f"({100.0 * len(kept) / total:.1f}%)")
    print(f"  dropped, gain <= 0 (flat/worsening): {len(drop_gain)}  "
          f"({100.0 * len(drop_gain) / total:.1f}%)  <- the filter's stated purpose")
    print(f"  dropped by MONOTONE alone, gain > 0: {len(drop_nonmono)}  "
          f"({100.0 * len(drop_nonmono) / total:.1f}%)  <- the question")
    print()
    if kept:
        print(f"  kept walls' gain:      median {statistics.median(kept):.1f}%  "
              f"max {max(kept):.1f}%")
    if drop_nonmono:
        gains = [g for g, _, _, _ in drop_nonmono]
        big = [d for d in drop_nonmono if d[0] >= BIG_GAIN_PCT]
        print(f"  non-monotone drops:    median {statistics.median(gains):.1f}%  "
              f"max {max(gains):.1f}%")
        print(f"  of those, gain >= {BIG_GAIN_PCT:.0f}%:  {len(big)}  "
              f"({100.0 * len(big) / total:.1f}% of all walls)")
        print()
        print("  the steepest walls rejected for a non-monotone tail:")
        for g, param, where, lats in sorted(drop_nonmono, reverse=True)[:8]:
            print(f"    {g:5.1f}%  {param:<16} {where:<34} tail {lats}")
    print()
    print("READ IT AS A COST, NOT A VERDICT. A non-monotone tail can be a real non-monotonicity in the")
    print("knob (in which case the filter is right) or one noisy middle point in a descending tail (in")
    print("which case a steep slope was discarded). This probe cannot tell those apart -- the per-trial")
    print("std on this project is 16%, so a single mid-tail spike is entirely affordable by noise. What")
    print("it CAN say is how much slope is sitting behind the predicate, which is what decides whether")
    print("the next round should look at it at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
