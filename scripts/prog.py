"""Read a run's events.jsonl from disk and report progress. Disk is the only source of truth."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
ev = run / "events.jsonl"
if not ev.exists():
    print("NO events.jsonl at", ev)
    sys.exit(1)

kinds = Counter()
first_ts = None
last_ts = None
best = None
best_cand = None
trials = 0
trials_ok = 0
rewrites = 0
agent_fail = 0
ended = None
reeval = []
families = {}
fam_status = {}
global_conv = None

with open(ev, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        t = e.get("type") or e.get("event_type") or "?"
        kinds[t] += 1
        ts = e.get("ts") or e.get("timestamp")
        if isinstance(ts, (int, float)):
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        p = e.get("payload") or {}
        if t == "TRIAL_DONE":
            trials += 1
            tr = p.get("trial") or p
            st = (tr.get("status") or "").lower()
            v = tr.get("latency_ms") or {}
            val = v.get("median") if isinstance(v, dict) else None
            if val is None and isinstance(v, dict):
                val = v.get("mean")
            if st == "complete" and isinstance(val, (int, float)):
                trials_ok += 1
                if best is None or val < best:
                    best = val
                    best_cand = tr.get("candidate_id")
        elif t in ("REWRITE_PRODUCED", "REWRITE_ROUND_DONE"):
            rewrites += 1
        elif t == "AGENT_CALL_FAILED":
            agent_fail += 1
        elif t == "RUN_FINISHED":
            ended = p
            # The final re-eval lives HERE, in RUN_FINISHED.payload.summary.best, written by
            # `_finalize`. There is no FINAL_REEVAL_DONE event: this branch used to key on that
            # name and so the `if reeval:` block below never printed, on any run, ever. Same bug
            # as check_wrapup.py's first draft, found by scripts/check_event_names.py.
            b = ((p.get("summary") or {}).get("best")) or {}
            if b:
                reeval.append({k: b.get(k) for k in (
                    "final_reeval_ms", "final_reeval_median_ms", "final_reeval_ok", "tuned_ms",
                    "precision", "excessive_speedup_flag", "candidate_id")})
        elif t == "CONVERGENCE_DECIDED":
            # Two DIFFERENT events share this name and they must not be merged:
            #   * per-family (orchestrator.py:2122) carries `family_id` and a real verdict;
            #   * global (orchestrator.py:606) carries NO family_id -- its
            #     `decision.evidence.families` is a snapshot of each family's STATUS
            #     (active/frozen_*), not a verdict, and its own verdict+stop_kind is how the run
            #     ended.
            # Reading `p["verdict"]`/`p["family_id"]` (one level too shallow) printed
            # `{fam-...: None}` for every family on every run. Merging the two into one dict then
            # let the global snapshot overwrite the per-family verdicts AND dropped
            # stop_kind=budget_exhausted, so verify both blocks on a finished run, not just one.
            dec = p.get("decision") or p
            fid = dec.get("family_id") or p.get("family_id")
            if fid:
                families[fid] = "%s/%s" % (dec.get("verdict"), dec.get("stop_kind"))
            else:
                global_conv = "%s/%s" % (dec.get("verdict"), dec.get("stop_kind"))
                fam_status = dict(((dec.get("evidence") or {}).get("families") or {}))

print("RUN:", run.name)
if first_ts and last_ts:
    print("elapsed_min: %.1f" % ((last_ts - first_ts) / 60.0))
    print("last_event_age_s: %.0f" % (time.time() - last_ts))
print("trials: %d (complete %d)" % (trials, trials_ok))
if best is not None:
    print("best_tuned_ms: %.4f  cand=%s" % (best, best_cand))
print("rewrite_events:", rewrites, " agent_failures:", agent_fail)
if families:
    print("family verdicts (verdict/stop_kind):", families)
if global_conv:
    print("global convergence:", global_conv, " family status at that point:", fam_status)
if reeval:
    print("FINAL_REEVAL:", json.dumps(reeval[-1])[:400])
if ended:
    print("ENDED:", json.dumps(ended)[:400])
else:
    print("ENDED: (no RUN_FINISHED -- still in flight or killed)")
print("--- event counts ---")
for k, v in kinds.most_common(25):
    print("  %-28s %d" % (k, v))
