"""Is a frozen best a PLATEAU (physics) or a stalled tuner? Two different situations.

Recorded distinction: on L3:48 a frozen best was physics -- 8 of 10 candidates across 4 families
landed inside 2%, all at 93.7-95.5% of the measured DRAM roof. But a frozen best can also mean the
tuner is resampling a narrow region. The separator is how many DISTINCT configs tie within the
noise floor: many means the limit is not the search.

THE THRESHOLD MUST BE RELATIVE TO SAMPLE SIZE. My first version used a fixed ">= 4 distinct
configs = plateau", which called both live runs a possible stall at 31 and 92 trials. Measured
against finished L3:43 runs, that was wrong in a way worth recording:

    corpus, 885 trials -> 4 distinct near-best   (0.45 per 100 trials)
    corpus, 709 trials -> 7 distinct near-best   (0.99 per 100 trials)
    live,    92 trials -> 2 distinct near-best   (2.17 per 100 trials)
    live,    31 trials -> 2 distinct near-best   (6.45 per 100 trials)

The live runs have a HIGHER near-best rate per trial than the finished ones, so 2-of-92 is an
early-sample effect, not a narrow peak. A fixed count would flag every young run -- and a warning
that fires on every run is not a warning. So the verdict is withheld below a minimum sample and the
rate is reported alongside the count.
"""
import json, pathlib, sys, collections, statistics
d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(d).name
rows = []
for l in open(d + "/events.jsonl", encoding="utf-8"):
    e = json.loads(l)
    if e["type"] != "TRIAL_DONE": continue
    tr = (e.get("payload") or {}).get("trial") or {}
    if tr.get("status") != "complete" or tr.get("failure_kind"): continue
    v = ((tr.get("params") or {}).get("values")) or {}
    m = (tr.get("latency_ms") or {}).get("median")
    if m: rows.append((m, tuple(sorted(v.items())), tr.get("candidate_id")))
if not rows:
    print("%s no completed trials" % tag); raise SystemExit
best = min(r[0] for r in rows)
near = [r for r in rows if r[0] <= best * 1.0235]     # inside the measured 2.35% noise floor
distinct = len({r[1] for r in near})
cands = collections.Counter(r[2] for r in near)
print("%s  completed %d | best %.4f ms" % (tag, len(rows), best))
print("   within the 2.35%% noise floor of best: %d trials, %d DISTINCT configs, %d candidate(s)" % (
    len(near), distinct, len(cands)))
rate = 100.0 * distinct / len(rows)
print("   near-best rate: %.2f distinct configs per 100 completed trials" % rate)
# Finished L3:43 runs sit at 0.45-0.99 per 100. A run BELOW that band, with enough trials to
# say so, is the narrow-peak case; at or above it the search is still finding new ties.
if len(rows) < 150:
    print("   => NO VERDICT: %d completed trials is too few. Finished L3:43 runs needed 700-885 "
          "to accumulate 4-7 distinct near-best configs" % len(rows))
else:
    # The 0.45-0.99 band comes from finished L3:43 runs. Applying it to a DIFFERENT task would be
    # the cross-task comparison this project has a recorded error for -- L3:21 reads 0.18 here,
    # which may be its own normal rather than a narrow peak. So the rate is reported against its
    # reference band and the band's provenance is stated; no verdict is asserted across tasks.
    print("   => %.2f per 100. Reference band from finished L3:43 runs: 0.45-0.99 (%s)." % (
        rate, "within/above it" if rate >= 0.40 else "BELOW it"))
    print("      That band is L3:43-derived; on another task compare against that task's own "
          "finished runs, not against this number.")
allms = sorted(r[0] for r in rows)
print("   latency spread: p10 %.4f  median %.4f  p90 %.4f  (max %.4f)" % (
    allms[len(allms)//10], statistics.median(allms), allms[int(len(allms)*0.9)], allms[-1]))
# which knobs actually vary among the near-best set
if distinct >= 2:
    keys = collections.defaultdict(set)
    for _, cfg, _ in near:
        for k, val in cfg: keys[k].add(val)
    varying = {k: sorted(v) for k, v in keys.items() if len(v) > 1}
    print("   knobs that VARY among near-best configs: %s" % (varying or "none -- all identical"))
