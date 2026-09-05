"""Running prospective check: how does the boundary fix behave on expansions it did NOT see?

The fix in `fix-boundary-direction-follows-the-winning-trial.md` was derived from a
retrospective audit of 1126 knobs. Everything since then is out-of-sample: the fix is
driver-side, so the run in flight still uses the OLD rule, and every expansion it performs is
a free test of what the new rule would have requested.

Replays the old and new predicates over every expansion in a run and reports, per expansion,
whether the requested knob set changes. Two numbers matter:

  * the CHANGED / INERT ratio, which should track the retrospective 22.4% / 77.6% split. If
    the fix turned out to alter most expansions, "mostly subtractive" would be wrong.
  * for each expansion, whether the winner used an added value -- the outcome the fix is
    ultimately aiming at.

Usage: python scripts/audit_boundary_fix_prospective.py [run-dir ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from kernel_optimizer.config import load_config          # noqa: E402
from kernel_optimizer.control.orchestrator import boundary_knobs_to_expand  # noqa: E402
from kernel_optimizer.models.core import ParameterSpace, TrialRecord        # noqa: E402
from kernel_optimizer.models.reports import TuningStats                     # noqa: E402
from kernel_optimizer.tuning.stats import TuningStatsAnalyzer               # noqa: E402

cfg = load_config(Path("configs/experiments_l3.yaml"))
analyzer = TuningStatsAnalyzer(cfg.device)

runs = [Path(a) for a in sys.argv[1:]] or sorted(Path("runs").glob("run-l3-*"))
changed = inert = 0
hit = miss = 0

for run in runs:
    f = run / "events.jsonl"
    if not f.exists():
        continue
    spaces, trials, old, best = [], defaultdict(list), {}, {}
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            spaces.append(ParameterSpace.model_validate(p["space"]))
        elif e["type"] == "TRIAL_DONE":
            t = TrialRecord.model_validate(p["trial"])
            trials[t.space_id].append(t)
        elif e["type"] == "STATS_DONE":
            old[p["stats"]["space_id"]] = p["stats"]
        elif e["type"] == "TUNING_DONE":
            best[p["space_id"]] = p["best_ms"]

    by_cand = defaultdict(list)
    for s in spaces:
        by_cand[s.candidate_id].append(s)
    printed_run = False
    for cid, vs in by_cand.items():
        vs.sort(key=lambda s: s.version)
        for prev, nxt in zip(vs, vs[1:]):
            if prev.space_id not in old or not trials.get(prev.space_id):
                continue
            o = TuningStats.model_validate(old[prev.space_id])
            n = analyzer.analyze(prev, trials[prev.space_id])
            ol = [(k["name"], k["direction"])
                  for k in boundary_knobs_to_expand(o, 0.8, prev,
                                                    min_effect_pct=cfg.budgets.min_improvement_pct)]
            nw = [(k["name"], k["direction"])
                  for k in boundary_knobs_to_expand(n, 0.8, prev,
                                                    min_effect_pct=cfg.budgets.min_improvement_pct)]
            verdict = "CHANGED" if ol != nw else "inert"
            changed += verdict == "CHANGED"
            inert += verdict == "inert"

            # did the post-expansion winner use a value the expansion added?
            oldc = {d.name: d.choices for d in prev.domains}
            added = {d.name: [c for c in d.choices if c not in oldc.get(d.name, d.choices)]
                     for d in nxt.domains}
            added = {k: v for k, v in added.items() if v}
            win = None
            for t in trials.get(nxt.space_id, []):
                if t.status == "complete" and t.latency_ms:
                    if win is None or t.latency_ms.mean < win.latency_ms.mean:
                        win = t
            used = bool(win and any(win.params.values.get(k) in v for k, v in added.items()))
            if win:
                hit += used
                miss += not used
            if not printed_run:
                print(f"\n{run.name}")
                printed_run = True
            b0, b1 = best.get(prev.space_id), best.get(nxt.space_id)
            print(f"  {cid} {verdict:8s} old={[k for k, _ in ol]} new={[k for k, _ in nw]}"
                  f"  {b0} -> {b1} ms  winner_used_added={used}")

tot = changed + inert
if tot:
    print(f"\nexpansions replayed: {tot}   CHANGED {changed} ({changed/tot*100:.1f}%)"
          f"   inert {inert} ({inert/tot*100:.1f}%)")
    print(f"retrospective expectation: 22.4% of knobs change verdict, 77.6% unchanged")
if hit + miss:
    print(f"winner used an added value: {hit} of {hit+miss} ({hit/(hit+miss)*100:.0f}%)")


# ---------------------------------------------------------------------------
# UNIT WARNING, because I got this wrong once and it matters for how the fix is described.
#
# The retrospective 22.4% is a PER-KNOB rate. This script reports a PER-EXPANSION rate, and
# an expansion changes if ANY knob in its requested set changes. With ~3 flagged knobs per
# expansion the per-knob rate compounds: 1-(1-0.224)^3 = 53%.
#
# So "mostly subtractive / mostly inert" is a true statement about KNOBS and a FALSE one
# about EXPANSIONS. Roughly half of expansions get a different knob set; what stays true is
# that within those sets the change is overwhelmingly a withdrawal, not a re-aiming.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PER-KNOB rate over the same spaces (the unit the 22.4% figure uses)")

nk = ck = withdrawn = raised = flipped = 0
for run in runs:
    f = run / "events.jsonl"
    if not f.exists():
        continue
    spaces, trials, old = [], defaultdict(list), {}
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            spaces.append(ParameterSpace.model_validate(p["space"]))
        elif e["type"] == "TRIAL_DONE":
            t = TrialRecord.model_validate(p["trial"])
            trials[t.space_id].append(t)
        elif e["type"] == "STATS_DONE":
            old[p["stats"]["space_id"]] = p["stats"]
    for s in spaces:
        if s.space_id not in old or not trials.get(s.space_id):
            continue
        o = TuningStats.model_validate(old[s.space_id])
        n = analyzer.analyze(s, trials[s.space_id])
        for a in o.param_stats:
            b = next((x for x in n.param_stats if x.name == a.name), None)
            if b is None:
                continue
            nk += 1
            if (a.at_boundary, a.boundary_direction) == (b.at_boundary, b.boundary_direction):
                continue
            ck += 1
            if a.at_boundary and not b.at_boundary:
                withdrawn += 1
            elif b.at_boundary and not a.at_boundary:
                raised += 1
            else:
                flipped += 1

if nk:
    print(f"  knobs: {nk}   changed {ck} ({ck/nk*100:.1f}%)"
          f"   withdrawn {withdrawn}   newly raised {raised}   re-aimed {flipped}")
    print(f"  compounding to a ~3-knob expansion set: "
          f"1-(1-{ck/nk:.3f})^3 = {(1-(1-ck/nk)**3)*100:.0f}% of expansions")
    print("\n  'Mostly inert' is true PER KNOB and false PER EXPANSION. Within a changed set the")
    print("  change is overwhelmingly a withdrawal rather than a re-aiming, which is the")
    print("  property that makes the fix conservative.")
