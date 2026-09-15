"""The true same-CODE A/A: one materialized artifact measured more than once.

The cross-arm 13 pairs fix the seed and the knobs but NOT the parameterized body, so their
spread is an upper bound. Within a single arm the same artifact hash can be timed more than
once (re-tunes of an identical config, and reused-measurement records). Those repeats are the
only same-code repeats these logs contain. Reused records are reported separately because a
reused record is NOT an independent measurement.
"""
import json,hashlib,collections,statistics
from pathlib import Path
base=Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v4")
TARGETS=[("n1-a","run-l3-43-20260915-071127"),("n1-b","run-l3-43-20260915-071127")]
allfresh=[]
for arm,RUN in TARGETS:
    d=base/arm/RUN
    by=collections.defaultdict(list)
    for line in (d/"events.jsonl").open(encoding="utf-8",errors="replace"):
        line=line.strip()
        if not line: continue
        try: e=json.loads(line)
        except ValueError: continue
        if e.get("type")!="TRIAL_DONE": continue
        p=e.get("payload") or {}; t=p.get("trial") or {}
        if t.get("status")!="complete": continue
        m=(t.get("latency_ms") or {}).get("median")
        if m is None: continue
        cid=str(t.get("candidate_id")); tid=str(t.get("trial_id"))
        f=d/"candidates"/cid/"trials"/(tid+".py")
        if not f.exists(): continue
        by[hashlib.sha256(f.read_bytes()).hexdigest()].append(
            (float(m), bool(p.get("reused_measurement")), tid, cid))
    rep={k:v for k,v in by.items() if len(v)>1}
    print("### %s : distinct artifacts %d ; artifacts timed >1x : %d"%(arm,len(by),len(rep)))
    fresh=[]; withreused=[]
    for k,v in sorted(rep.items()):
        ms=[x[0] for x in v]; nr=sum(1 for x in v if x[1])
        spread=(max(ms)-min(ms))/min(ms)*100.0
        (fresh if nr==0 else withreused).append(spread)
        if len(fresh)+len(withreused)<=12:
            print("   %s  n=%d reused=%d  ms=%s  spread=%.2f%%"%(
                k[:10],len(v),nr,",".join("%.4f"%x for x in ms),spread))
    for name,arr in (("FRESH repeats only (no reused record)",fresh),("with >=1 reused record",withreused)):
        if arr:
            s=sorted(arr)
            print("   %s: n=%d  min %.2f%%  median %.2f%%  p90 %.2f%%  max %.2f%%"%(
                name,len(s),s[0],statistics.median(s),s[int(0.9*(len(s)-1))],s[-1]))
        else:
            print("   %s: n=0"%name)
    allfresh+=fresh
    print()
if allfresh:
    s=sorted(allfresh)
    print("=== POOLED same-code fresh-repeat A/A across both arms (same box, same task) ===")
    print("  n=%d  min %.2f%%  median %.2f%%  p90 %.2f%%  max %.2f%%"%(
        len(s),s[0],statistics.median(s),s[int(0.9*(len(s)-1))],s[-1]))
    print("  This is same-code, same-box, fresh-vs-fresh: the closest thing in these logs to a")
    print("  pure timer noise floor. It is a WITHIN-ARM figure; a cross-arm claim needs the")
    print("  declaration-matched upper bound instead.")
else:
    print("=== no same-code fresh repeat exists in these logs: pure timer noise is UNMEASURED ===")
