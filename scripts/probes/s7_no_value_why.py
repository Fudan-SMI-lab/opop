"""Which wall reached `_toward_wall`, and why was there no value left toward it?

WHAT THIS CORRECTS. I reported the treatment arm's zeros as "no wall" throughout. The skip counters for
its SECOND candidate say otherwise:

    n_told=10  no_wall=1  no_value=0
    n_told=20  no_wall=2  no_value=1   <- a wall WAS found here
    n_told=30  no_wall=3  no_value=1
    n_told=40  no_wall=4  no_value=2   <- and here

`n_skipped_no_value_toward_wall` increments in `_walls`, AFTER a wall survived both `find_walls` and
`select_for_probing`, when `_toward_wall` returns None. So on 2 of 4 recomputes the mechanism had a
probe-worthy wall in hand and still enqueued nothing. That is a different fact from "no wall existed" and
it is the one place the counters point at S7's OWN logic rather than at the wall criteria.

`_toward_wall` returns None for three reasons, and they are not equally interesting:
  * every choice toward the wall is at or beyond the REFUSED value -- nothing launchable is left. The
    space's top choice IS the refused value, so the wall is at the edge of the declared domain.
  * the only candidate value equals the incumbent's own value -- the anchor, which exists so the log
    says "nothing left toward the wall" rather than "already asked".
  * the knob is missing from the space or the incumbent, so no base point can be formed.

Which one it is decides whether anything could be done about it. This prints the arithmetic per wall:
the domain, the incumbent's value, the refused value, and the choices between them.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/s7_no_value_why.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/work/opop/src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.evaluation import soft_wall as soft_wall_mod  # noqa: E402
from kernel_optimizer.evaluation import wall_attribution  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    LatencyStats, ParamDomain, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
)
from kernel_optimizer.tuning.slope_guide import SlopeGuide, _drawn_values  # noqa: E402
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer  # noqa: E402

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
CFG = Path("/root/autodl-tmp/work/opop/configs/experiments_s7_treatment_box4gpu0.yaml")


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
    runs = sorted(BASE.glob("s7-treatment/run-l3-43-*"))
    if not runs:
        print("no treatment run")
        return 2
    ev = _events(runs[-1])
    cfg = load_config(CFG)
    sg = cfg.v3.slope_guide
    soft_on = sg.use_soft_wall and cfg.v3.soft_wall.enabled
    analyzer = TuningStatsAnalyzer(cfg.device)
    spaces, trials = _spaces(ev), _trials(ev)

    for p in [(e.get("payload") or {}) for e in ev if e.get("type") == "SLOPE_GUIDE_STEP"]:
        if not int(p.get("n_skipped_no_value_toward_wall") or 0):
            continue
        sid, n_told = str(p.get("space_id") or ""), int(p.get("n_told") or 0)
        space = spaces.get(sid)
        if space is None:
            continue
        own = [t for t in trials if t.space_id == sid][:n_told]
        stats = analyzer.analyze(space, own)
        refused = [t.params.values for t in own
                   if t.failure_kind == "infeasible_shared_memory" and t.params]
        guide = SlopeGuide(space=space, recompute_every=sg.recompute_every,
                           max_enqueued_per_recompute=sg.max_enqueued_per_recompute,
                           use_soft_wall=soft_on)
        incumbent = guide._incumbent(own)
        drawn = _drawn_values(own)

        print(f"=== space {sid}  n_told={n_told}  "
              f"no_value={p.get('n_skipped_no_value_toward_wall')}")
        hard = wall_attribution.find_walls(stats, refused) if refused else []
        worth, _ = wall_attribution.select_for_probing(hard, -1) if hard else ([], [])
        soft = soft_wall_mod.find_soft_walls(stats, own) if soft_on else None
        rows: list[tuple[str, str, float, str, float | None]] = []
        for w in worth:
            rows.append((w.param, w.side, w.tail_gain_pct, "hard_wall", float(w.refused_value)))
        for swall in (soft.walls if soft else []):
            rows.append((swall.param, "high", swall.tail_gain_pct, "soft_wall", None))
        if not rows:
            print("    (no probe-worthy wall in this replay -- counters came from the live run)")
            continue
        for knob, side, gain, source, refv in sorted(rows, key=lambda r: r[2], reverse=True):
            dom = next((d for d in space.domains if d.name == knob), None)
            choices = list(dom.choices) if dom else []
            inc = (incumbent or {}).get(knob)
            got = guide._toward_wall(knob, side, drawn.get(knob, set()), refv, inc)
            print(f"    {source:<10} {knob:<16} side={side:<5} gain={gain:5.1f}%")
            print(f"      declared choices : {choices}")
            print(f"      incumbent value  : {inc!r}")
            print(f"      refused value    : {refv!r}")
            print(f"      _toward_wall ->    {got!r}")
            if got is None:
                if refv is not None and choices and _num(choices[-1]) == refv:
                    print("      REASON: the domain's TOP choice IS the refused value -- the wall sits")
                    print("              at the edge of the declared space, so nothing launchable is")
                    print("              left toward it. Not fixable by sampling; only a wider space")
                    print("              or a rewrite can move it.")
                else:
                    print("      REASON: every choice toward the wall is the incumbent's own value or")
                    print("              beyond the refusal bound (see _toward_wall's three rules).")
    return 0


def _num(x: object) -> float | None:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
