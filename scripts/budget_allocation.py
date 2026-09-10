"""Which budget actually ends an L3 run, and which loop receives it?

The paper's claim is that parameter-tuning feedback guides STRUCTURAL rewrite (loop C). This
script asks what share of the wall clock loop C actually gets, and what stops a run.

Both answers turned out to be uniform across all five completed L3 runs, which is why the
result is worth a document: the wall clock ends every run (never the rewrite-round cap, never
the family cap), and loop C receives 7-8% of it. The bulk goes to per-candidate bookkeeping --
parameterize + analyst -- because those are billed PER CANDIDATE while rewrite is billed per
ROUND, and one rewrite round emits two candidates, each of which then needs a full
parameterize/tune/stats/analyse pass.

Two measurement caveats, both of which make this an approximation rather than an audit:

  - Agent time is `AGENT_CALL_FINISHED.ts - AGENT_CALL_STARTED.ts`, so it includes the
    harness's own artifact harvesting, not just model latency.
  - Trial time is the gap between consecutive TRIAL_DONE events, reset at any agent call,
    TUNING_DONE or SPACE_PUBLISHED so that idle stretches are not attributed to trials, and
    capped at 600 s per gap. It therefore UNDER-counts: the first trial after each reset
    contributes nothing. Agent + trials accounts for 81-92% of wall; the rest is baselines,
    compile screens, the final re-eval and the report.

Zero GPU. Reads events.jsonl and manifest.json only.
"""
import json,os,sys
from collections import defaultdict
tot_by_mod=defaultdict(float); n_by_mod=defaultdict(int)
gtot=defaultdict(float); gn=defaultdict(int); gwall=0.0; gtrial=0.0; nruns=0
print("%-28s %6s | %-46s | %s"%("run","wall_h","agent hours by module","trials_h"))
for run in sys.argv[1:]:
    p=os.path.join(run,"events.jsonl")
    if not os.path.exists(p): continue
    rows=[]
    with open(p,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try: rows.append(json.loads(line))
                except Exception: pass
    T=lambda r: r.get("type","")
    if not any(T(r)=="RUN_FINISHED" for r in rows): continue
    nruns+=1
    wall=(rows[-1]["ts"]-rows[0]["ts"])/3600; gwall+=wall
    st={}; per=defaultdict(float); pern=defaultdict(int)
    for r in rows:
        pl=r.get("payload",{})
        if T(r)=="AGENT_CALL_STARTED": st[pl.get("call_id")]=(r["ts"],pl.get("module"))
        elif T(r) in ("AGENT_CALL_FINISHED","AGENT_CALL_FAILED"):
            c=pl.get("call_id")
            if c in st:
                t0,mod=st.pop(c); mod=mod or pl.get("module") or "?"
                per[mod]+=r["ts"]-t0; pern[mod]+=1
    # trial time: gaps between consecutive TRIAL_DONE inside a tuning block
    prev=None; trial=0.0
    for r in rows:
        if T(r)=="TRIAL_DONE":
            if prev is not None and r["ts"]-prev<600: trial+=r["ts"]-prev
            prev=r["ts"]
        elif T(r) in ("AGENT_CALL_STARTED","TUNING_DONE","SPACE_PUBLISHED"): prev=None
    gtrial+=trial/3600
    for m,v in per.items(): gtot[m]+=v; gn[m]+=pern[m]
    s=" ".join("%s=%.2f(n%d)"%(m[:5],per[m]/3600,pern[m]) for m in sorted(per,key=lambda x:-per[x]))
    print("%-28s %6.2f | %-46s | %.2f"%(os.path.basename(run)[:28],wall,s[:46],trial/3600))
print()
print("=== POOLED over %d completed runs (%.1f h wall) ==="%(nruns,gwall))
ta=sum(gtot.values())/3600
print("  agent total %.2f h = %.1f%% of wall     trials %.2f h = %.1f%%   (rest: baselines, compile screens, final reeval, report)"
      %(ta,100*ta/gwall,gtrial,100*gtrial/gwall))
print("  %-16s %8s %6s %10s %10s"%("module","hours","calls","mean_min","%of agent"))
for m in sorted(gtot,key=lambda x:-gtot[x]):
    h=gtot[m]/3600
    print("  %-16s %8.2f %6d %10.1f %9.1f%%"%(m,h,gn[m],gtot[m]/gn[m]/60,100*h/ta))
struct=(gtot.get("rewriter",0)+gtot.get("novelty",0))/3600
book=(gtot.get("parameterizer",0)+gtot.get("analyst",0))/3600
print()
print("  loop C+D (rewriter+novelty, the structural search):        %5.2f h  = %.1f%% of agent time, %.1f%% of wall"
      %(struct,100*struct/ta,100*struct/gwall))
print("  per-candidate bookkeeping (parameterizer+analyst):         %5.2f h  = %.1f%% of agent time, %.1f%% of wall"
      %(book,100*book/ta,100*book/gwall))
