"""Format selected run events as one-line monitor notifications (stdin -> stdout)."""
import json
import os
import sys

WATCH = {
    "RUN_CREATED", "RUN_FINISHED", "RUN_FAILED", "BASELINE_DONE", "SEMANTICS_PROBED",
    "TUNING_DONE", "SPACE_REJECTED", "SPACE_EXPANSION_REJECTED", "SPACE_EXPANDED",
    "FAMILY_ROUND_RECORDED", "CONVERGENCE_DECIDED", "AGENT_CALL_FAILED",
}
run = os.environ.get("RF", "")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    t = e.get("type")
    if t not in WATCH:
        continue
    p = e.get("payload") or {}
    bits = []
    if t == "BASELINE_DONE" and "baseline" in p:
        b = p["baseline"]
        bits = ["%s=%s" % (b["kind"], b["latency_ms"]["mean"])]
    elif t == "CONVERGENCE_DECIDED" and "decision" in p:
        d = p["decision"]
        bits = [str(d.get("scope")), str(d.get("verdict")), str(d.get("stop_kind") or "")]
        if p.get("family_id"):
            bits.append(p["family_id"])
    elif t == "RUN_FINISHED":
        best = (p.get("summary") or {}).get("best")
        bits = ["BEST=%s ms" % (best.get("tuned_ms") if best else None)]
        if best:
            bits.append("reeval_ok=%s" % best.get("final_reeval_ok"))
    elif t == "SEMANTICS_PROBED":
        s = p.get("semantics") or {}
        bits = ["training=%s norm_layers=%d" % (s.get("training"), len(s.get("norm_layers") or []))]
    else:
        for k in ("candidate_id", "family_id", "best_ms", "round", "reason", "module", "error"):
            if k in p:
                bits.append("%s=%s" % (k, str(p[k])[:70]))
    print("%s %s %s" % (run, t, " ".join(bits)), flush=True)
