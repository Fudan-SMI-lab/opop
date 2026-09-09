"""Do the trials we ALREADY PAID FOR contain the resource-impact knowledge the agent lacks?

The user's third blind spot: the agent does not know what its change will do to resource
usage. But every tuning trial already records (params -> shared_bytes, n_regs, n_spills,
num_warps, latency). So for KNOB changes, the mapping "change -> resource impact" is already
on disk, unused.

This measures how much is recoverable:
  1. For each knob, how strongly does it move each resource? (a per-knob sensitivity table)
  2. Is that mapping CONSISTENT across candidates, or candidate-specific? -- decides whether
     it can be given to an agent as general advice or only as candidate-local advice.
  3. Would knowing it have avoided real waste? Cross-check against the shared-memory failures.

If (1) and (2) come out well, then part of blind spot 3 needs NO new measurement machinery --
only reading what we already write.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = [Path(a) for a in sys.argv[1:]]

# (run, cand, knob) -> {value -> [(shared, regs, spills, warps, ms)]}
obs = defaultdict(lambda: defaultdict(list))
fail_by_knobval = defaultdict(lambda: [0, 0])

for run in runs:
    ev = run / "events.jsonl"
    if not ev.exists():
        continue
    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            vals = (tr.get("params") or {}).get("values") or {}
            cid = tr.get("candidate_id")
            ok = (tr.get("status") or "").lower() == "complete"
            fk = tr.get("failure_kind")
            for k, v in vals.items():
                slot = fail_by_knobval[(k, str(v))]
                slot[0 if ok else 1] += 1
            if not ok:
                continue
            pr = tr.get("profile") or {}
            sh, rg = pr.get("shared_bytes"), pr.get("n_regs")
            sp, wp = pr.get("n_spills"), pr.get("num_warps")
            ms = (tr.get("latency_ms") or {}).get("median")
            if sh is None and rg is None:
                continue
            for k, v in vals.items():
                obs[(run.name, cid, k)][str(v)].append((sh, rg, sp, wp, ms))

print("=" * 104)
print("1. PER-KNOB RESOURCE SENSITIVITY -- already on disk, currently unused")
print("=" * 104)
print("For each knob: does changing it move shared memory / registers, and by how much?")
print("%-16s %-20s %5s %14s %14s %12s" %
      ("knob", "candidate", "vals", "shared range", "regs range", "ms range"))

knob_effect = defaultdict(list)   # knob -> list of (shared_span, regs_span)
rows = 0
for (rname, cid, knob), byval in sorted(obs.items()):
    if len(byval) < 2:
        continue
    sh_med, rg_med, ms_med = {}, {}, {}
    for v, recs in byval.items():
        shs = sorted(r[0] for r in recs if r[0] is not None)
        rgs = sorted(r[1] for r in recs if r[1] is not None)
        mss = sorted(r[4] for r in recs if r[4] is not None)
        if shs:
            sh_med[v] = shs[len(shs) // 2]
        if rgs:
            rg_med[v] = rgs[len(rgs) // 2]
        if mss:
            ms_med[v] = mss[len(mss) // 2]
    sh_span = (max(sh_med.values()) - min(sh_med.values())) if len(sh_med) > 1 else 0
    rg_span = (max(rg_med.values()) - min(rg_med.values())) if len(rg_med) > 1 else 0
    ms_span = ((max(ms_med.values()) / min(ms_med.values()) - 1) * 100
               if len(ms_med) > 1 and min(ms_med.values()) > 0 else 0)
    knob_effect[knob].append((sh_span, rg_span))
    if rows < 26 and (sh_span or rg_span):
        print("%-16s %-20s %5d %14s %14s %11.1f%%"
              % (knob, cid, len(byval), "%+d B" % sh_span, "%+d" % rg_span, ms_span))
        rows += 1

print()
print("=" * 104)
print("2. IS THE KNOB->RESOURCE MAPPING CONSISTENT ACROSS CANDIDATES?")
print("=" * 104)
print("(decides: general advice to the agent, or candidate-local only)")
print("%-18s %8s %16s %16s %s" % ("knob", "seen in", "median shared span", "median regs span",
                                  "consistent?"))
for knob, effs in sorted(knob_effect.items(), key=lambda kv: -len(kv[1])):
    if len(effs) < 2:
        continue
    shs = sorted(e[0] for e in effs)
    rgs = sorted(e[1] for e in effs)
    msh, mrg = shs[len(shs) // 2], rgs[len(rgs) // 2]
    # consistent if every candidate agrees on WHETHER this knob moves the resource
    moves_sh = [e[0] > 0 for e in effs]
    moves_rg = [e[1] > 0 for e in effs]
    agree = ("shared: %s" % ("all" if all(moves_sh) else "none" if not any(moves_sh) else "MIXED")
             + ", regs: %s" % ("all" if all(moves_rg) else "none" if not any(moves_rg)
                               else "MIXED"))
    print("%-18s %8d %16s %16s %s" % (knob, len(effs), "%d B" % msh, "%d" % mrg, agree))

print()
print("=" * 104)
print("3. WOULD THIS HAVE AVOIDED REAL WASTE? knob values with the worst failure rates")
print("=" * 104)
print("%-18s %-12s %8s %8s %s" % ("knob", "value", "passed", "failed", "fail rate"))
worst = sorted(((k, v, ok, bad) for (k, v), (ok, bad) in fail_by_knobval.items()
                if ok + bad >= 8), key=lambda r: -(r[3] / (r[2] + r[3])))[:14]
for k, v, ok, bad in worst:
    print("%-18s %-12s %8d %8d %8.0f%%" % (k, v, ok, bad, 100 * bad / (ok + bad)))
print()
print("A knob value with a high failure rate AND a large resource span is exactly what an")
print("agent should be told BEFORE it writes a rewrite that leans on that value.")
