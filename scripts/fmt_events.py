"""Format selected run events as one-line monitor notifications (stdin -> stdout).

WATCH is checked against the event types the harness actually emits (grep `store.append(`
in src/). Two were out of step and are corrected here:

  * KERNELS_NEVER_LAUNCHED was emitted but not watched. It is the deterministic dead-code
    detector -- a @triton.jit kernel defined but never launched across a WHOLE tuning
    budget, i.e. the budget measured something other than the advertised optimization.
    That failure is silent by construction: correctness passes, timing looks normal, no
    error appears (L3:21 cand-c0b3b7cd spent 31 trials timing an eager fallback). It has
    never fired on disk, so watching it costs nothing and its first occurrence is exactly
    the kind of thing that must not scroll past unnoticed.
  * NOVELTY_REJECTED was emitted but not watched: a rewrite or novelty candidate discarded
    for a duplicate structural signature. Silent budget loss -- an agent call paid for,
    nothing registered.
  * RUN_FAILED was watched but is emitted nowhere; kept as a harmless forward declaration
    and marked as such so nobody reads its silence as a healthy signal.
"""
import json
import os
import sys

WATCH = {
    "RUN_CREATED", "RUN_FINISHED", "BASELINE_DONE", "SEMANTICS_PROBED",
    "TUNING_DONE", "SPACE_REJECTED", "SPACE_EXPANSION_REJECTED", "SPACE_EXPANDED",
    "FAMILY_ROUND_RECORDED", "CONVERGENCE_DECIDED", "AGENT_CALL_FAILED",
    # silent-failure signals: emitted by the harness, previously unwatched
    "KERNELS_NEVER_LAUNCHED", "NOVELTY_REJECTED",
    "RUN_FAILED",  # not emitted by any current code path; forward declaration only
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
    elif t == "KERNELS_NEVER_LAUNCHED":
        # The whole point is WHICH kernel was dead and how much budget it consumed.
        bits = ["%s never_launched=%s over %s trials" % (
            p.get("candidate_id"), ",".join(p.get("never_launched") or []),
            p.get("n_trials_measured"))]
    else:
        # `origin` and `never_launched` are carried here for NOVELTY_REJECTED and any
        # future event whose payload uses them.
        for k in ("candidate_id", "family_id", "best_ms", "round", "origin", "reason",
                  "module", "error"):
            if k in p:
                bits.append("%s=%s" % (k, str(p[k])[:70]))
    print("%s %s %s" % (run, t, " ".join(bits)), flush=True)
