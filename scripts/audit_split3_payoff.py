"""Does split3 ever WIN? The contract change's payoff, measured rather than assumed.

split3 costs 3x the MMAs, so on a short reduction or a wide dynamic range it is pure loss -- which
is exactly why it was made a KNOB rather than a recommendation. The honest question is therefore
not "is split3 good" but "does the tuner have a real choice": both values reachable, both measured,
and the winner decided on latency.

Compared WITHIN each (COMPUTE_DTYPE) so the comparison is like-for-like: split3 at bf16 against
plain at bf16, never against plain at fp16.
"""
import json, pathlib, sys, collections
d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(d).name
best = collections.defaultdict(dict)
for l in open(d + "/events.jsonl", encoding="utf-8"):
    e = json.loads(l)
    if e["type"] != "TRIAL_DONE": continue
    tr = (e.get("payload") or {}).get("trial") or {}
    if tr.get("status") != "complete" or tr.get("failure_kind"): continue
    v = ((tr.get("params") or {}).get("values")) or {}
    m = (tr.get("latency_ms") or {}).get("median")
    if not m: continue
    p, mode = v.get("COMPUTE_DTYPE"), v.get("DOT_MODE")
    cur = best[p].get(mode)
    if cur is None or m < cur: best[p][mode] = m
print("%s  best latency per (precision, DOT_MODE):" % tag)
wins = 0; comparable = 0
for p in sorted(best, key=str):
    row = best[p]
    pl, sp = row.get("plain"), row.get("split3")
    if pl and sp:
        comparable += 1
        d_pct = 100 * (sp - pl) / pl
        if sp < pl: wins += 1
        print("   %-5s plain %.4f  split3 %.4f  => split3 is %+.2f%% %s" % (
            p, pl, sp, d_pct, "WINS" if sp < pl else ""))
    else:
        print("   %-5s plain %-9s split3 %-9s  (not comparable)" % (
            p, round(pl, 4) if pl else "-", round(sp, 4) if sp else "-"))
print("   => split3 wins %d of %d like-for-like precision comparisons" % (wins, comparable))
