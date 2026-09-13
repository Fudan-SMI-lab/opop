import json
import sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")


def latest(arm: str) -> Path | None:
    ds = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    return ds[-1] if ds else None


for arm in ("s7-treatment", "s7-control"):
    R = latest(arm)
    print(f"=== {arm}  {R.name if R else '(none)'}")
    if R is None:
        continue
    ev = []
    with (R / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass

    def pl(e):
        return e.get("payload") or e

    # Trials per candidate, with the completed count -- S7's cadence is on trials TOLD.
    per = {}
    for e in ev:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = pl(e).get("trial") or {}
        cid = t.get("candidate_id") or "?"
        d = per.setdefault(cid, {"n": 0, "ok": 0, "refused": 0})
        d["n"] += 1
        if t.get("status") == "complete":
            d["ok"] += 1
        if t.get("failure_kind") == "infeasible_shared_memory":
            d["refused"] += 1
    for cid, d in per.items():
        print(f"    {cid}: trials={d['n']} complete={d['ok']} refused_sharedmem={d['refused']}")

    steps = [pl(e) for e in ev if e.get("type") == "SLOPE_GUIDE_STEP"]
    print(f"    SLOPE_GUIDE_STEP events: {len(steps)}")
    for p in steps:
        enq = p.get("enqueued") or []
        ref = p.get("refused") or []
        print(f"      n_told={p.get('n_told')}/{p.get('budget')} "
              f"cand={str(p.get('candidate_id'))[:14]} "
              f"enq={len(enq)} refused={len(ref)} "
              f"recomputes={p.get('n_recomputes')} suggested={p.get('n_suggested')}")
        print(f"        skips: no_wall={p.get('n_skipped_no_wall')} "
              f"no_value={p.get('n_skipped_no_value_toward_wall')} "
              f"already={p.get('n_skipped_already_proposed')} "
              f"bad_incumbent={p.get('n_skipped_incomplete_incumbent')}")
        for row in enq:
            print(f"        ENQ {row.get('knob')}={row.get('knob_value')} "
                  f"src={row.get('source')} gain={row.get('tail_gain_pct')}%")
        for row in ref:
            print(f"        REF {row.get('knob')}={row.get('knob_value')} "
                  f"why={row.get('refused')}")
    fails = [pl(e) for e in ev if e.get("type") == "SLOPE_GUIDE_FAILED"]
    for p in fails:
        print(f"    !! SLOPE_GUIDE_FAILED: {p.get('error')}")
    # Any orchestrator-level failure worth catching early.
    for typ in ("RESOURCE_WALL_ATTRIBUTION_FAILED", "RESOURCE_SOFT_WALL_FAILED",
                "AGENT_CALL_FAILED", "TRIAL_ARTIFACT_FAILED"):
        n = sum(1 for e in ev if e.get("type") == typ)
        if n:
            print(f"    !! {typ}: {n}")
    print(f"    TUNING_DONE: {sum(1 for e in ev if e.get('type') == 'TUNING_DONE')}")
    for e in ev:
        if e.get("type") == "TUNING_DONE":
            p = pl(e)
            print(f"      slope_guide={p.get('slope_guide')}")
            break
sys.stdout.flush()
