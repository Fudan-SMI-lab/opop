"""Cause C: does the REAL SlopeGuide, replayed offline, agree with what the live run logged?

WHY THIS IS THE LAST OPEN QUESTION. S7 has fired 7 times on the live treatment arm and enqueued nothing.
Three causes were possible and two are now measured and closed: walls get erased by a later successful
trial, and `monotone` rejects 28.1% of walls (mostly correctly). The third was never tested --

  C. the in-loop path behaves differently from the offline replay that priced this mechanism at 17.1%.

`s7_no_wall_why.py` re-derives the wall stages with `find_walls` / `select_for_probing`, which shows what
SHOULD have happened. It does NOT run `SlopeGuide` itself, so it cannot detect a defect in the class that
sits between them -- the incumbent resolution, the `_base_in_space` check, the `_toward_wall` target
choice, the dedup. Any of those could silently swallow a wall that both stage functions agreed existed.

THE TEST. Instantiate the SHIPPING `SlopeGuide` with each space's own config, feed it that space's trials
truncated at exactly the live recompute points, and compare its counters field-by-field against the
`SLOPE_GUIDE_STEP` payloads the run wrote. Agreement means the class does in the replay what it did in
the run, so the offline reasoning about it transfers. DISAGREEMENT IS A DEFECT and is actionable.

This is the positive-control-shaped part: the comparison can FAIL LOUDLY. A probe that merely printed the
replay's counters would agree with itself no matter what the run did.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/s7_replay_vs_log.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/work/opop/src")

from kernel_optimizer.config import load_config  # noqa: E402
from kernel_optimizer.models.core import (  # noqa: E402
    LatencyStats, ParamDomain, ParameterSpace, ParamSet, ProfileRecord, TrialRecord,
)
from kernel_optimizer.tuning.slope_guide import SlopeGuide  # noqa: E402
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer  # noqa: E402

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
CFG_DIR = Path("/root/autodl-tmp/work/opop/configs")

# The counters that must match. Deliberately the SKIP counters and not just the totals: two runs can
# both enqueue 0 for entirely different reasons, and "0 == 0" would hide a class that reached that zero
# down the wrong branch.
FIELDS = ("n_recomputes", "n_suggested", "n_skipped_no_wall",
          "n_skipped_no_value_toward_wall", "n_skipped_already_proposed",
          "n_skipped_incomplete_incumbent", "n_proposed_never_drawn_value")


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
    arm, cfg_name = "s7-treatment", "experiments_s7_treatment_box4gpu0.yaml"
    runs = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    if not runs:
        print("no treatment run")
        return 2
    run = runs[-1]
    ev = _events(run)
    cfg = load_config(CFG_DIR / cfg_name)
    sg_cfg = cfg.v3.slope_guide
    analyzer = TuningStatsAnalyzer(cfg.device)
    spaces, trials = _spaces(ev), _trials(ev)

    logged = [(e.get("payload") or {}) for e in ev if e.get("type") == "SLOPE_GUIDE_STEP"]
    print(f"=== {arm}  {run.name}")
    print(f"    config: recompute_every={sg_cfg.recompute_every} "
          f"max_enqueued={sg_cfg.max_enqueued_per_recompute} "
          f"use_soft_wall={sg_cfg.use_soft_wall and cfg.v3.soft_wall.enabled}")
    print(f"    SLOPE_GUIDE_STEP events logged: {len(logged)}")
    if not logged:
        print("    nothing to compare yet")
        return 0

    mismatches = 0
    # One guide per SPACE, exactly as the orchestrator builds it, replayed in log order so the
    # cumulative counters accumulate the way the live ones did.
    guides: dict[str, SlopeGuide] = {}
    for p in logged:
        sid = str(p.get("space_id") or "")
        space = spaces.get(sid)
        if space is None:
            print(f"    !! space {sid} not declared -- cannot replay")
            mismatches += 1
            continue
        g = guides.get(sid)
        if g is None:
            g = SlopeGuide(space=space, recompute_every=sg_cfg.recompute_every,
                           max_enqueued_per_recompute=sg_cfg.max_enqueued_per_recompute,
                           use_soft_wall=(sg_cfg.use_soft_wall and cfg.v3.soft_wall.enabled))
            guides[sid] = g

        n_told = int(p.get("n_told") or 0)
        own = [t for t in trials if t.space_id == sid][:n_told]
        stats = analyzer.analyze(space, own)
        # `measured_keys` is the tuner's drawn set; the params of the trials that ran ARE those keys.
        drawn = {t.params.key() for t in own if t.params}
        sug = g.suggest(stats, own, drawn)
        snap = g.snapshot()

        bad = [f for f in FIELDS if snap.get(f) != p.get(f)]
        n_enq_logged = len(p.get("enqueued") or [])
        if len(sug) != n_enq_logged + len(p.get("refused") or []):
            bad.append(f"suggestions {len(sug)} vs logged "
                       f"{n_enq_logged}+{len(p.get('refused') or [])}")
        status = "OK" if not bad else "MISMATCH"
        print(f"    n_told={n_told:<4} space={sid} {status}")
        if bad:
            mismatches += 1
            for f in bad:
                if f in FIELDS:
                    print(f"        {f}: replay {snap.get(f)!r} vs log {p.get(f)!r}")
                else:
                    print(f"        {f}")

    print()
    if mismatches:
        print(f"DISAGREEMENT on {mismatches} of {len(logged)} recomputes -- THIS IS A DEFECT. The class")
        print("does not do offline what it did in the run, so every offline conclusion about S7 is")
        print("suspect and this is actionable regardless of the pair's outcome.")
        return 1
    print(f"AGREEMENT on all {len(logged)} recomputes, field by field including the skip counters.")
    print("So cause C is CLOSED: the in-loop path behaves as the offline replay does, and S7's zeros are")
    print("the two measured applicability limits (walls erased by later successes; monotone rejecting")
    print("28.1% of walls, mostly correctly) rather than a defect in the mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
