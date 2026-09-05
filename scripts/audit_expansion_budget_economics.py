"""What does improvement K's expansion COST, and how often does the re-tune pay for it?

`audit_expansion_direction_yield.py` asks whether the post-expansion best USED an added value.
This asks the budget question instead, which is what decides whether K is worth its slot:

    an expansion buys a whole fresh tuning budget (trials_per_space, 40) for one candidate.
    How often does that 40-trial re-tune actually lower the candidate's best latency, and by
    how much -- against the alternative of spending those 40 trials on a rewrite round?

Prompted by two live expansions in run-l3-21-20260905-195615 that landed on opposite sides:
  cand-47371017  9.78 -> 9.42 ms  (improved, but via pre-existing values)
  cand-faa8862d  15.1 -> 15.1 ms  (no improvement; the added BLOCK_N=256 ran 17.4 and failed once)

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
            spaces.append((s["space_id"], s["candidate_id"], s["version"],
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
    by_cand = defaultdict(list)
    for sid, cid, ver, doms in d["spaces"]:
        by_cand[cid].append((ver, sid, doms))
    best_of = {sid: ms for sid, _, ms in d["tunings"]}
    for cid, versions in by_cand.items():
        versions.sort()
        for (v0, s0, dom0), (v1, s1, dom1) in zip(versions, versions[1:]):
            if s0 not in best_of or s1 not in best_of:
                continue
            added = {k: [c for c in v if c not in dom0.get(k, v)] for k, v in dom1.items()}
            added = {k: v for k, v in added.items() if v}
            if not added:
                continue
            before, after = best_of[s0], best_of[s1]
            rows.append({
                "run": d["run"], "cand": cid, "before": before, "after": after,
                "gain_pct": (before - after) / before * 100.0 if before else 0.0,
                "trials": d["trials"].get(s1, 0), "knobs": len(added),
                "added": added,
            })

if not rows:
    print("no expansions found")
    raise SystemExit(0)

improved = [r for r in rows if r["after"] < r["before"] - 1e-9]
flat = [r for r in rows if abs(r["after"] - r["before"]) <= 1e-9]
worse = [r for r in rows if r["after"] > r["before"] + 1e-9]
spent = sum(r["trials"] for r in rows)

print(f"expansions with a before/after tuning pair: {len(rows)}")
print(f"  improved the candidate's best : {len(improved):3d}  ({len(improved)/len(rows)*100:.0f}%)")
print(f"  left it unchanged             : {len(flat):3d}  ({len(flat)/len(rows)*100:.0f}%)")
print(f"  ended WORSE                   : {len(worse):3d}  ({len(worse)/len(rows)*100:.0f}%)")
print(f"\ntrials spent on expansion re-tunes: {spent}"
      f"  (median {statistics.median([r['trials'] for r in rows]):.0f} per expansion)")

if improved:
    gains = [r["gain_pct"] for r in improved]
    print(f"\nwhen it improved: median {statistics.median(gains):.1f}%  "
          f"mean {statistics.mean(gains):.1f}%  max {max(gains):.1f}%")
    print("  largest gains:")
    for r in sorted(improved, key=lambda r: -r["gain_pct"])[:6]:
        print(f"    {r['run'][8:]} {r['cand']}  {r['before']:.2f} -> {r['after']:.2f} ms"
              f"  ({r['gain_pct']:.1f}%, {r['trials']} trials)")

# The comparison that matters: a rewrite round costs a comparable budget.
print("\n=== against the alternative: what a REWRITE round delivered, same runs")
rw = []
for run in sorted(Path("runs").glob("run-l3-*")):
    d = scan(run)
    if not d:
        continue
    # first tuning per candidate, grouped by family via candidate order of appearance
    firsts = {}
    for sid, cid, ms in d["tunings"]:
        firsts.setdefault(cid, ms)
    seq = list(firsts.values())
    for a, b in zip(seq, seq[1:]):
        if a and b:
            rw.append((b - a) / a * -100.0)
if rw:
    wins = [g for g in rw if g > 0]
    print(f"  candidate-to-candidate steps: {len(rw)}, improving {len(wins)} "
          f"({len(wins)/len(rw)*100:.0f}%), median gain when improving "
          f"{statistics.median(wins):.1f}%")
    print("  (a coarse proxy -- consecutive candidates are not all rewrites of each other --")
    print("   quoted only to place the expansion numbers on a scale, not as a controlled test)")


# ---------------------------------------------------------------------------
# THE DECISIVE SPLIT: is the gain from the ADDED VALUES, or just from 40 more trials?
#
# 68% of expansions improve the candidate, but audit_expansion_direction_yield.py shows
# only 12-16% end with the best USING an added value. If the improvement rate is the same
# whether or not an added value won, then what K buys is a fresh tuning budget -- which a
# plain re-tune of the unchanged space would also buy, without an agent call.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("DECISIVE SPLIT: gain from added values, or from the extra trials?")

used, notused = [], []
for run in sorted(Path("runs").glob("run-l3-*")):
    d = scan(run)
    if not d:
        continue
    by_cand = defaultdict(list)
    for sid, cid, ver, doms in d["spaces"]:
        by_cand[cid].append((ver, sid, doms))
    best_of = {sid: ms for sid, _, ms in d["tunings"]}
    # winning param values per space, from the fastest complete trial
    win = {}
    f = run / "events.jsonl"
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        t = e["payload"]["trial"]
        lat = (t.get("latency_ms") or {}).get("mean")
        if t.get("status") == "complete" and lat:
            cur = win.get(t["space_id"])
            if cur is None or lat < cur[0]:
                win[t["space_id"]] = (lat, t["params"]["values"])
    for cid, versions in by_cand.items():
        versions.sort()
        for (v0, s0, dom0), (v1, s1, dom1) in zip(versions, versions[1:]):
            if s0 not in best_of or s1 not in best_of or s1 not in win:
                continue
            added = {k: [c for c in v if c not in dom0.get(k, v)] for k, v in dom1.items()}
            added = {k: v for k, v in added.items() if v}
            if not added:
                continue
            wv = win[s1][1]
            hit = any(wv.get(k) in vals for k, vals in added.items())
            rec = {"before": best_of[s0], "after": best_of[s1], "cand": cid,
                   "run": d["run"],
                   "gain": (best_of[s0] - best_of[s1]) / best_of[s0] * 100.0}
            (used if hit else notused).append(rec)

for label, group in (("best USED an added value", used),
                     ("best used NO added value", notused)):
    if not group:
        continue
    imp = [r for r in group if r["after"] < r["before"] - 1e-9]
    gains = [r["gain"] for r in imp]
    print(f"  {label:26s} n={len(group):3d}  improved {len(imp):3d} "
          f"({len(imp)/len(group)*100:3.0f}%)  median gain "
          f"{statistics.median(gains) if gains else 0:.1f}%")

print("\nIf those two rows are close, K's benefit is the extra tuning budget rather than the")
print("widened range -- and a plain re-tune would deliver it without an agent call. That is a")
print("BUDGET POLICY question (re-tune vs expand vs rewrite), not a defect: nothing here is")
print("wrong, and changing it alters what every candidate gets. Recorded, not acted on.")
