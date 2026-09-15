"""For each of the 13 declaration-matched pairs: is the EXECUTED code byte-identical?

The original generator's key was (registered seed source_sha, declared knob dict). That fixes
the SEED and the KNOBS. It does not fix the PARAMETERIZED body -- the parameterizer is an LLM
call, so the two arms can turn one seed into two different programs while both keep the seed's
journalled source_sha. This script prices that: per pair, the two materialized artifacts' hashes
and, when they differ, how many source lines differ.
"""
import json,hashlib,collections,statistics,difflib
from pathlib import Path
base=Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v4"); RUN="run-l3-43-20260915-071127"
per={}
for arm in ("n1-a","n1-b"):
    d=base/arm/RUN
    ev=[json.loads(l) for l in (d/"events.jsonl").open(encoding="utf-8",errors="replace") if l.strip()]
    sha={}
    for e in ev:
        if e.get("type")=="CANDIDATE_REGISTERED":
            c=(e.get("payload") or {}).get("candidate") or {}
            if c.get("origin")=="seed" and c.get("source_sha"): sha[str(c["candidate_id"])]=c["source_sha"][:12]
    dd=collections.defaultdict(list)
    for e in ev:
        if e.get("type")!="TRIAL_DONE": continue
        p=e.get("payload") or {}; t=p.get("trial") or {}
        if t.get("status")!="complete": continue
        cid=str(t.get("candidate_id"))
        if cid not in sha: continue
        m=(t.get("latency_ms") or {}).get("median")
        if m is None: continue
        tid=str(t.get("trial_id"))
        f=d/"candidates"/cid/"trials"/(tid+".py")
        art=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
        dd[(sha[cid], json.dumps((t.get("params") or {}).get("values") or {},sort_keys=True))].append(
            dict(median=float(m), tid=tid, cid=cid, art=art, path=f,
                 reused=bool(p.get("reused_measurement")), space=str(t.get("space_id"))))
    per[arm]=dd
shared=sorted(set(per["n1-a"])&set(per["n1-b"]))
print("declaration-matched pairs: %d"%len(shared))
print()
hdr=("#","seed","A_trial","B_trial","A_art","B_art","code","A_ms","B_ms","signed%")
print("  %-3s %-13s %-12s %-12s %-11s %-11s %-9s %10s %10s %9s"%hdr)
identical=0; diffs=[]
for i,k in enumerate(shared,1):
    ra=min(per["n1-a"][k],key=lambda x:x["median"]); rb=min(per["n1-b"][k],key=lambda x:x["median"])
    same = (ra["art"] is not None and ra["art"]==rb["art"])
    identical+=same
    dv=(rb["median"]-ra["median"])/ra["median"]*100.0; diffs.append(abs(dv))
    print("  %-3d %-13s %-12s %-12s %-11s %-11s %-9s %10.4f %10.4f %+8.2f%%"%(
        i,k[0],ra["tid"],rb["tid"],(ra["art"] or "-")[:10],(rb["art"] or "-")[:10],
        "SAME" if same else "DIFFERENT", ra["median"], rb["median"], dv))
print()
print("  executed code byte-identical in %d / %d pairs"%(identical,len(shared)))
print()
print("=== per-seed: do the two arms' parameterized bodies differ at all? ===")
seeds=sorted({k[0] for k in shared})
for s in seeds:
    ka=[k for k in per["n1-a"] if k[0]==s]; kb=[k for k in per["n1-b"] if k[0]==s]
    fa=per["n1-a"][ka[0]][0]["path"]; fb=per["n1-b"][kb[0]][0]["path"]
    ta=fa.read_text(encoding="utf-8",errors="replace").splitlines()
    tb=fb.read_text(encoding="utf-8",errors="replace").splitlines()
    dl=[l for l in difflib.unified_diff(ta,tb,lineterm="",n=0) if l[:1] in "+-" and l[:3] not in ("+++","---")]
    print("  seed %s : %d differing source lines between arms (A %d lines, B %d lines)"%(s,len(dl),len(ta),len(tb)))
print()
print("=== consequence for the noise-floor label ===")
s=sorted(diffs)
print("  spread over the 13 declaration-matched pairs: median %.2f%% p90 %.2f%% max %.2f%%"%(
    statistics.median(s), s[int(0.9*(len(s)-1))], s[-1]))
print("  This spread MIXES timer noise with parameterization differences, so it is an UPPER")
print("  BOUND on same-code measurement noise, not a measurement of it.")
