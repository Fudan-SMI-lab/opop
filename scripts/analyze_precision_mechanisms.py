"""Separate two populations I had conflated: ONLY-bf16 failers vs fails-at-every-precision.

The `ieee fails at 7.59x` row came from cand-277a15ac alone -- a candidate that ALSO fails at
fp16/tf32. Pooling it with the others made the error look dtype-independent. The right partition:

  A) fails at EVERY precision incl. ieee  => a plain algorithm bug; precision is irrelevant.
  B) fails ONLY at bf16, passes fp16+tf32 (both 10 mantissa bits) and ieee
     => cannot be RANGE (bf16 has fp16's beaten, 8 exponent bits vs 5)
     => cannot be a code path (fp16 and bf16 take the SAME branch in these sources)
     => a 7-vs-10 mantissa-bit shortfall: cause (a), exactly what split3 addresses.
"""
import json, pathlib, re, sys, collections
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../external_files/box2-runs/runs-l3")
A = []; B = []; other = []
for d in sorted(root.glob("run-l3-*")):
    per = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for l in open(d / "events.jsonl", encoding="utf-8"):
        e = json.loads(l)
        if e["type"] != "TRIAL_DONE": continue
        tr = (e.get("payload") or {}).get("trial") or {}
        if tr.get("failure_kind") in ("infeasible_shared_memory", "runtime_error"): continue
        p = (((tr.get("params") or {}).get("values")) or {}).get("COMPUTE_DTYPE")
        ok = tr.get("status") == "complete" and not tr.get("failure_kind")
        per[tr.get("candidate_id")][p]["ok" if ok else "bad"] += 1
    for cid, byp in sorted(per.items()):
        def rate(p):
            c = byp.get(p)
            return (c["ok"] / sum(c.values()), sum(c.values())) if c else (None, 0)
        bf, nbf = rate("bf16"); f16, nf16 = rate("fp16"); t32, nt32 = rate("tf32"); ie, nie = rate("ieee")
        if nbf < 3 or f16 is None: continue
        row = (d.name[-8:], cid[:13], "%d" % nbf, "%.0f%%" % (100*bf),
               "%.0f%%/%d" % (100*f16, nf16),
               ("%.0f%%/%d" % (100*t32, nt32)) if t32 is not None else "-",
               ("%.0f%%/%d" % (100*ie, nie)) if ie is not None else "-")
        clean_hi = f16 >= 0.95 and (t32 is None or t32 >= 0.95) and (ie is None or ie >= 0.95)
        if bf <= 0.05 and clean_hi: B.append(row)
        elif bf <= 0.05: A.append(row)
        else: other.append(row)
hdr = "%-9s %-14s %-5s %-6s %-9s %-9s %-9s" % ("run","candidate","n_bf16","bf16 ok","fp16","tf32","ieee")
print("B) ONLY bf16 fails, every 10-bit+ precision passes >=95%%  -- cause (a), split3's target")
print("   " + hdr)
for r in B: print("   " + "%-9s %-14s %-5s %-6s %-9s %-9s %-9s" % r)
print("\nA) bf16 fails AND at least one 10-bit+ precision also degraded -- algorithm bug, not precision")
print("   " + hdr)
for r in A: print("   " + "%-9s %-14s %-5s %-6s %-9s %-9s %-9s" % r)
print("\nother (bf16 passes):  %d candidates" % len(other))
nb = sum(int(r[2]) for r in B)
print("\n=> population B: %d candidates, %d bf16 trials, ZERO passes, while fp16/tf32 (same 10\n"
      "   mantissa bits) and ieee pass. Range is excluded: bf16 has 8 exponent bits vs fp16's 5." % (len(B), nb))
