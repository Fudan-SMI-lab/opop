"""Field-by-field coverage of TRIAL_DONE.profile -- the audit the offline rescoring needs first.

WHY THIS COMES BEFORE ANY RESCORING. `profile` is optional on TrialRecord, TRIAL_DONE also
carries failures, and a record may be a REUSED measurement rather than an independent one.
So "the schema can express peak_alloc_bytes" is not "every trial has it", and a rescoring
that quietly drops the records missing a field would select on the very thing it measures.

Reports per run, and never pools: separate columns for complete/failed, for reused vs
independently measured, and per precision -- because a memory objective compared across
fp16 and bf16 lineages is not "the same task with a different objective".

Missing is printed as missing. It is never filled with zero.
"""
import json, sys, collections
from pathlib import Path

FIELDS = ["peak_alloc_bytes", "peak_reserved_bytes", "peak_above_resident_bytes",
          "candidate_aten_bytes", "candidate_aten_ops", "threads_launched",
          "n_regs", "n_spills", "shared_bytes", "occupancy", "sass",
          "compile_s", "wall_ms", "cpu_issue_ms", "overhead_gpu_ms"]

def main():
    for arg in sys.argv[1:]:
        run = Path(arg.rstrip("/"))
        rows, reused, status, prec, backend = [], collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        have = collections.Counter(); nonnull = collections.Counter()
        for line in (run / "events.jsonl").open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line: continue
            try: e = json.loads(line)
            except ValueError: continue
            if e.get("type") != "TRIAL_DONE": continue
            p = e.get("payload") or {}
            t = p.get("trial") or {}
            rows.append(t)
            reused["reused" if p.get("reused_measurement") else "independent"] += 1
            status[str(t.get("status"))] += 1
            v = (t.get("params") or {}).get("values") or {}
            prec[str(v.get("COMPUTE_DTYPE", v.get("DOT_PRECISION", "n/a")))] += 1
            pr = t.get("profile")
            if pr is None:
                have["<no profile at all>"] += 1
                continue
            for f in FIELDS:
                if f in pr: have[f] += 1
                if pr.get(f) is not None: nonnull[f] += 1
        n = len(rows)
        print("=" * 92)
        print("%s   (%s)   TRIAL_DONE = %d" % (run.parent.name, run.name, n))
        if not n: continue
        print("  status:    %s" % dict(status))
        print("  provenance:%s   <-- reused records are NOT independent measurements" % dict(reused))
        print("  precision: %s" % dict(prec))
        print("  no profile object at all: %d" % have["<no profile at all>"])
        print("\n  %-28s %8s %8s   %8s" % ("field", "present", "non-null", "usable%"))
        for f in FIELDS:
            print("  %-28s %8d %8d   %7.1f%%" % (f, have[f], nonnull[f], 100.0 * nonnull[f] / n))
        # The subset a memory objective could actually be scored on: complete, independent,
        # and carrying the field.
        ok = 0
        for t in rows:
            pr = t.get("profile") or {}
            if t.get("status") == "complete" and pr.get("peak_alloc_bytes") is not None:
                ok += 1
            
        print("\n  rescorable on peak_alloc_bytes (complete AND non-null): %d/%d = %.1f%%"
              % (ok, n, 100.0 * ok / n))
        print("  => any cross-objective champion must be read INSIDE this subset, and the")
        print("     excluded records reported, or the comparison selects on availability.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
