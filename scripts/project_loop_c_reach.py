"""Will this run reach Loop C (rewrite rounds) inside its wall clock? G27's check depends on it.

Recorded: 5 of 5 completed L3 runs were ended by WALL CLOCK, and they used only 1-2 of 5 rewrite
rounds. So "will it get there" is a real question, not a formality -- and the answer is only useful
BEFORE the budget is spent.

Loop C starts after every seed candidate has been parameterized and tuned. Projects from measured
rates rather than assumed ones.
"""
import json, sys, time
d = sys.argv[1]; tag = sys.argv[2]
ev = [json.loads(l) for l in open(d + "/events.jsonl", encoding="utf-8")]
t0 = ev[0]["ts"]; now = ev[-1]["ts"]
elapsed = (now - t0) / 3600
man = json.load(open(d + "/manifest.json", encoding="utf-8"))
b = (man.get("config") or {}).get("budgets") or {}
wall = float(b.get("wall_clock_hours") or 12)
per_space = int(b.get("trials_per_space") or 40)
seeds = int(b.get("max_seed_candidates") or 4)
n_tri = sum(1 for e in ev if e["type"] == "TRIAL_DONE")
n_cand = sum(1 for e in ev if e["type"] == "CANDIDATE_REGISTERED")
n_pub = sum(1 for e in ev if e["type"] == "SPACE_PUBLISHED")
n_btl = sum(1 for e in ev if e["type"] == "BOTTLENECK_CLASSIFIED")
n_round = sum(1 for e in ev if e["type"] == "FAMILY_ROUND_RECORDED")
# agent time is per-candidate in the seed loop; measure what one candidate has actually cost
starts = {}; agent = 0.0
for e in ev:
    p = e.get("payload") or {}; m = p.get("module") or ""
    if e["type"] == "AGENT_CALL_STARTED": starts[m] = e["ts"]
    if e["type"] == "AGENT_CALL_FINISHED" and m in starts: agent += e["ts"] - starts.pop(m)
done_cands = n_btl                      # a candidate is finished once it is classified
print("%s  elapsed %.2f h of %.0f h | trials %d | candidates done %d/%d | rewrite rounds %d" % (
    tag, elapsed, wall, n_tri, done_cands, seeds, n_round))
if done_cands >= 1:
    per_cand_h = elapsed / done_cands
    remaining_seed_h = per_cand_h * (seeds - done_cands)
    print("    measured %.2f h per finished candidate => %.2f h to finish the seed loop" % (
        per_cand_h, remaining_seed_h))
    left_for_C = wall - elapsed - remaining_seed_h
    print("    projected wall clock left for Loop C: %.2f h" % left_for_C)
    if left_for_C <= 0:
        print("    => WILL NOT REACH Loop C: conversion stays 0-of-0 and G27 gets no evidence")
    else:
        # a rewrite round costs one rewriter call plus a full tuning pass on the survivor
        print("    => reaches Loop C with %.1f h, roughly %.1f rounds at %.2f h/candidate-pass" % (
            left_for_C, left_for_C / per_cand_h, per_cand_h))
else:
    # 0 finished candidates is EXACTLY the at-risk case, so a projection that only works after
    # one finishes is useless when it matters. Bound it from the per-trial gap instead.
    gaps = []
    tri = [e["ts"] for e in ev if e["type"] == "TRIAL_DONE"]
    for i in range(len(tri) - 1):
        gaps.append(tri[i + 1] - tri[i])
    if gaps:
        gaps.sort()
        g = gaps[len(gaps) // 2]
        tuning_h = per_space * g / 3600
        # agent cost so far is dominated by this candidate; treat it as the per-candidate figure
        per_cand = (agent / 3600) + tuning_h
        seed_h = seeds * per_cand
        print("    no candidate finished yet; bounding from the measured %.1f s/trial gap:" % g)
        print("      tuning %.2f h + agent %.2f h = %.2f h per candidate => %.2f h seed loop" % (
            tuning_h, agent / 3600, per_cand, seed_h))
        print("    => %s Loop C (%.2f h of %.0f h would remain)" % (
            "REACHES" if wall - seed_h > 0 else "WILL NOT REACH", wall - seed_h, wall))
    else:
        print("    no trials yet: nothing to project from")
print("    agent time so far %.2f h (%.0f%% of elapsed)" % (agent/3600, 100*agent/3600/elapsed))
