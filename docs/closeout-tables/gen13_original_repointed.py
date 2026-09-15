"""The ORIGINAL 13-pair generator, re-pointed at the archive. Logic byte-for-byte
unchanged from docs/reply-window1-offline-data-request.md 4.1: same matching key
(registered seed source_sha[:12], sorted declared knob dict), same min-of-medians
aggregation, same (B-A)/A denominator, same floor-index p90 with no interpolation,
same no-dedup and no-reused-handling. The ONLY edit is the base path.
"""
import json,collections,statistics,sys
base=sys.argv[1] + "/runs/%s/events.jsonl"
per={}
for a in ("n1-a","n1-b"):
    ev=[json.loads(l) for l in open(base%a,encoding="utf-8",errors="replace") if l.strip()]
    sha={}
    for e in ev:
        if e.get("type")=="CANDIDATE_REGISTERED":
            c=(e.get("payload") or {}).get("candidate") or {}
            if c.get("origin")=="seed" and c.get("source_sha"): sha[str(c["candidate_id"])]=c["source_sha"][:12]
    d=collections.defaultdict(list)
    for e in ev:
        if e.get("type")!="TRIAL_DONE": continue
        t=e["payload"]["trial"]
        if t.get("status")!="complete": continue
        cid=str(t.get("candidate_id"))
        if cid not in sha: continue
        m=(t.get("latency_ms") or {}).get("median")
        if m is None: continue
        key=(sha[cid], json.dumps((t.get("params") or {}).get("values") or {},sort_keys=True))
        d[key].append(float(m))
    per[a]=d
shared=sorted(set(per["n1-a"])&set(per["n1-b"]))
print("configs measured in BOTH arms (same seed sha AND byte-identical knob values): %d"%len(shared))
print()
diffs=[]
for k in shared:
    A=min(per["n1-a"][k]); B=min(per["n1-b"][k])
    dv=(B-A)/A*100; diffs.append(abs(dv))
    if len(diffs)<=14:
        print("  %s  A=%.4f (n=%d)  B=%.4f (n=%d)  B-A=%+6.2f%%"%(k[0],A,len(per["n1-a"][k]),B,len(per["n1-b"][k]),dv))
if diffs:
    diffs.sort()
    print()
    print("  TRUE same-config A/A |B-A|: n=%d  min %.2f%%  median %.2f%%  p90 %.2f%%  max %.2f%%"
          %(len(diffs),diffs[0],statistics.median(diffs),diffs[int(0.9*(len(diffs)-1))],diffs[-1]))
    print("  (compare: the tuned-best-per-seed spread was 10.85%%, but those were DIFFERENT configs)")
