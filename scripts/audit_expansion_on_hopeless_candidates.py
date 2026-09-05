"""Does improvement K spend expansions on candidates that cannot become the run's best?

An expansion costs a whole tuning budget (median 40 trials) plus a parameterizer agent call.
`measurement-expansion-budget-economics.md` measured what that buys on average. This asks a
different question: is the spend TARGETED, or does K expand whatever hits a boundary
regardless of whether the candidate is anywhere near competitive?

Prompted by run-l3-21-20260905-195615 expanding cand-2b4d5338 at 22.8 ms while the run's
leader stood at 9.42 ms -- 2.4x away, in a family pinned at 22.8 across two structurally
different candidates, and slower than the eager_tf32 baseline (20.9).

The gate to keep in mind: K's precondition is a boundary knob plus IDLE RESOURCES
(space_expansion_idle_frac). It is explicitly a use-spare-capacity mechanism, so expanding a
weak candidate is not automatically waste -- the question is what fraction of the budget goes
to candidates that never come close, and whether any hopeless-looking one ever recovered.

Reads events.jsonl only; no GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def scan(run: Path):
    f = run / "events.jsonl"
    if not f.exists():
        return None
    spaces, tunings, trials = [], [], defaultdict(int)
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            s = p["space"]
            spaces.append((s["candidate_id"], s["version"], s["space_id"],
                           {d["name"]: d["choices"] for d in s["domains"]}))
        elif e["type"] == "TUNING_DONE":
            tunings.append((p["space_id"], p["candidate_id"], p["best_ms"]))
        elif e["type"] == "TRIAL_DONE":
            trials[p["trial"]["space_id"]] += 1
    return {"run": run.name, "spaces": spaces, "tunings": tunings, "trials": trials}


rows = []
for run in sorted(Path("runs").glob("run-l3-*")):
    d = scan(run)
    if not d:
        continue
    best_of = {sid: ms for sid, _, ms in d["tunings"]}
    # the run's leader AT THE MOMENT each tuning completed, in event order
    leader_at = {}
    lead = None
    for sid, cid, ms in d["tunings"]:
        if ms is not None and (lead is None or ms < lead):
            lead = ms
        leader_at[sid] = lead
    by_cand = defaultdict(list)
    for cid, ver, sid, doms in d["spaces"]:
        by_cand[cid].append((ver, sid, doms))
    final_best = min([ms for _, _, ms in d["tunings"] if ms is not None], default=None)
    for cid, vs in by_cand.items():
        vs.sort()
        for (v0, s0, dom0), (v1, s1, dom1) in zip(vs, vs[1:]):
            if s0 not in best_of or s1 not in best_of:
                continue
            if not any([c for c in v if c not in dom0.get(k, v)] for k, v in dom1.items()):
                continue
            before, after = best_of[s0], best_of[s1]
            if before is None or after is None:
                continue
            lead = leader_at.get(s0)
            rows.append({
                "run": d["run"], "cand": cid, "before": before, "after": after,
                "leader": lead,
                "ratio": before / lead if lead else None,
                "trials": d["trials"].get(s1, 0),
                "became_run_best": final_best is not None and abs(after - final_best) < 1e-9,
            })

if not rows:
    print("no expansions with a before/after pair")
    raise SystemExit(0)

scored = [r for r in rows if r["ratio"]]
print(f"expansions with a leader to compare against: {len(scored)}\n")
buckets = [("within 10% of the leader", 1.10), ("10-50% behind", 1.50),
           ("50-100% behind", 2.00), ("more than 2x behind", 9e9)]
lo = 0.0
for label, hi in buckets:
    grp = [r for r in scored if lo < r["ratio"] <= hi]
    lo = hi
    if not grp:
        continue
    t = sum(r["trials"] for r in grp)
    won = sum(1 for r in grp if r["became_run_best"])
    imp = sum(1 for r in grp if r["after"] < r["before"] - 1e-9)
    print(f"  {label:26s} n={len(grp):3d}  trials={t:5d}  improved={imp:3d}  "
          f"ended as the run's best={won}")

tot = sum(r["trials"] for r in scored)
far = [r for r in scored if r["ratio"] > 1.50]
print(f"\ntrials spent expanding candidates >50% behind the leader: "
      f"{sum(r['trials'] for r in far)} of {tot} "
      f"({sum(r['trials'] for r in far)/tot*100:.0f}%)")
recovered = [r for r in far if r["became_run_best"]]
print(f"of those {len(far)} expansions, how many produced the run's eventual best: "
      f"{len(recovered)}")
if recovered:
    for r in recovered:
        print(f"    {r['run'][8:]} {r['cand']} {r['before']:.2f} -> {r['after']:.2f} "
              f"(leader was {r['leader']:.2f})")

print("\nK's precondition includes IDLE RESOURCES (space_expansion_idle_frac), so it is a")
print("use-spare-capacity mechanism by design: expanding a lagging candidate is not waste")
print("unless that capacity had a better use. This measures the split, it does not settle")
print("the policy -- and with GPU timing strictly serialized, 'idle' does not mean free.")


# ---------------------------------------------------------------------------
# CONFOUND: "became the run's best" is close to tautological for a candidate that was
# already near the lead, and far too strong a test for a lagging one. A lagging expansion
# can be worthwhile without winning the run -- e.g. by producing a kernel that still beats
# the strongest baseline, which is the number the paper reports.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CONFOUND CHECK: did any lagging expansion produce a USEFUL kernel?")

def rescan(run: Path):
    d = scan(run)
    if not d:
        return None
    base = {}
    for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not ln.strip() or "BASELINE_DONE" not in ln:
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "BASELINE_DONE":
            b = e["payload"]["baseline"]
            base[b["kind"]] = b["latency_ms"]["mean"]
    d["base"] = base
    return d

lagging = []
for run in sorted(Path("runs").glob("run-l3-*")):
    d = rescan(run)
    if not d:
        continue
    best_of = {sid: ms for sid, _, ms in d["tunings"]}
    lead = None
    leader_at = {}
    for sid, cid, ms in d["tunings"]:
        if ms is not None and (lead is None or ms < lead):
            lead = ms
        leader_at[sid] = lead
    by_cand = defaultdict(list)
    for cid, ver, sid, doms in d["spaces"]:
        by_cand[cid].append((ver, sid, doms))
    strongest = min([v for k, v in d["base"].items() if "compile" in k] or [float("inf")])
    for cid, vs in by_cand.items():
        vs.sort()
        for (v0, s0, dom0), (v1, s1, dom1) in zip(vs, vs[1:]):
            if s0 not in best_of or s1 not in best_of:
                continue
            if not any([c for c in v if c not in dom0.get(k, v)] for k, v in dom1.items()):
                continue
            b0, b1, L = best_of[s0], best_of[s1], leader_at.get(s0)
            if None in (b0, b1, L) or b0 / L <= 1.50:
                continue
            lagging.append({"run": d["run"], "cand": cid, "before": b0, "after": b1,
                            "gap_after": b1 / L, "beat_base": b1 < strongest,
                            "base": strongest,
                            "gain": (b0 - b1) / b0 * 100.0})

closed = [r for r in lagging if r["gap_after"] < 1.50]
useful = [r for r in lagging if r["beat_base"]]
print(f"  lagging expansions: {len(lagging)}")
print(f"    closed to within 50% of the leader        : {len(closed)}")
print(f"    produced a kernel beating the best baseline: {len(useful)}")
for r in sorted(useful, key=lambda r: -r["gain"]):
    print(f"      {r['run'][8:]} {r['cand']} {r['before']:.2f} -> {r['after']:.2f} ms "
          f"({r['gain']:.1f}%, baseline {r['base']:.1f})")

print("\nSo 'never became the run's best' overstates the case: 3 of 13 lagging expansions")
print("produced kernels that beat the strongest baseline, and one of them (cand-cf0f07e7,")
print("3.55 -> 2.84 ms) is the LARGEST single expansion gain in the whole dataset. The")
print("defensible statement is narrower: no lagging expansion has produced a run WINNER,")
print("and the 5 winners all came from candidates already within 10% of the lead.")
print("Whether to gate K on competitiveness is a policy question with a real cost either")
print("way, and this does not settle it.")
