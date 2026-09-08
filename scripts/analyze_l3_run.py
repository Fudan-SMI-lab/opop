"""End-of-run analysis for an L3 run. Answers the questions recorded in
docs/plan-next-round-and-deferred-fixes.md, from events.jsonl only.

Written BEFORE the run ended, so the questions are fixed in advance rather than chosen to suit
whatever the result turned out to be. Everything here reads the on-disk log -- never a summary,
never a notification.

Usage: python analyze_l3_run.py <run_dir>
"""

import collections
import json
import sys


def load(run_dir):
    return [json.loads(l) for l in open(run_dir + "/events.jsonl", encoding="utf-8")]


def main(run_dir):
    ev = load(run_dir)
    types = collections.Counter(e["type"] for e in ev)
    t0, tn = ev[0]["ts"], ev[-1]["ts"]

    print("=" * 78)
    print(f"RUN: {run_dir.rstrip('/').split('/')[-1]}")
    print(f"events {len(ev)}   elapsed {(tn - t0) / 3600:.2f} h   last event: {ev[-1]['type']}")

    # --- 1. How did it end? -------------------------------------------------------------
    print("\n--- 1. HOW IT ENDED " + "-" * 57)
    fin = next((e for e in reversed(ev) if e["type"] == "RUN_FINISHED"), None)
    if fin is None:
        print("  NO RUN_FINISHED: the run is still going, or it died. Check the process and the")
        print("  last event's timestamp before concluding anything -- a report tool will happily")
        print("  call an in-flight run 'killed or crashed'.")
    else:
        s = fin["payload"].get("summary") or fin["payload"]
        print(f"  stop_kind: {s.get('stop_kind')}   elapsed_hours: {s.get('elapsed_hours')}")
        gv = [e["payload"]["decision"] for e in ev
              if e["type"] == "CONVERGENCE_DECIDED"
              and (e["payload"].get("decision") or {}).get("scope") == "global"]
        if gv:
            print(f"  final global verdict: {gv[-1].get('verdict')} / {gv[-1].get('stop_kind')}")

    # --- 2. The number that counts ------------------------------------------------------
    print("\n--- 2. RESULT (final_reeval_ms, NOT tuned_ms) " + "-" * 32)
    if fin:
        s = fin["payload"].get("summary") or fin["payload"]
        best = s.get("best") or {}
        tuned, reeval = best.get("tuned_ms"), best.get("final_reeval_ms")
        print(f"  candidate: {best.get('candidate_id')}  family: {best.get('family_id')}")
        print(f"  tuned_ms        {tuned}")
        print(f"  final_reeval_ms {reeval}   <-- the publishable number")
        if tuned and reeval:
            gap = 100 * (reeval - tuned) / tuned
            print(f"  reeval gap: {gap:+.2f}%")
            # The gap is a TENDENCY, not a law. Prior runs had tuned optimistic by 1.5-6.7%, and
            # run-l1-42-20260908-023039 is a counterexample: its re-eval came out 2.65% FASTER
            # than the tuned figure. So the rule to follow is "quote final_reeval_ms", not
            # "assume tuned is inflated by roughly N%".
            print(f"    ({'re-eval slower than tuned, the usual direction' if gap > 0 else
                        're-eval FASTER than tuned -- the tendency is not a law'})")
        print(f"  speedups        {json.dumps(best.get('speedups'))}")
        print(f"  speedups_median {json.dumps(best.get('speedups_median'))}")
        note = best.get("speedups_median_note")
        if note:
            print(f"  median note: {note[:150]}")
        hv = best.get("honest_verdict") or s.get("honest_verdict")
        print(f"  honest_verdict  {json.dumps(hv)}")

    # --- 3. Did the rewrite rounds run, and did they improve? ---------------------------
    print("\n--- 3. LOOP C: ROUNDS AND GAINS " + "-" * 45)
    hist = collections.defaultdict(list)
    for e in ev:
        if e["type"] == "FAMILY_ROUND_RECORDED":
            hist[e["payload"]["family_id"]].append(e["payload"]["best_ms"])
    ne = collections.Counter(e["payload"]["family_id"] for e in ev
                             if e["type"] == "FAMILY_ROUND_NOT_EVALUATED")
    print(f"  rounds recorded (evaluated): {types['FAMILY_ROUND_RECORDED']}")
    print(f"  rounds that evaluated NOTHING: {types['FAMILY_ROUND_NOT_EVALUATED']} {dict(ne)}")
    print(f"  rewrites produced: {types['REWRITE_PRODUCED']}   "
          f"refused as duplicates: {types['REWRITE_REJECTED'] + types['NOVELTY_REJECTED']}"
          f"  (NOTE: older logs record rewrite refusals as NOVELTY_REJECTED)")
    for fid, h in hist.items():
        gains = [f"{100 * (h[i - 1] - h[i]) / h[i - 1]:+.1f}%" for i in range(1, len(h))]
        print(f"    {fid}: {[round(x, 3) for x in h]}  gains {gains or '(single round)'}")

    # --- 4. Did Loop D fire? ------------------------------------------------------------
    print("\n--- 4. LOOP D (novelty) " + "-" * 53)
    print(f"  NOVELTY_ROUND_STARTED {types['NOVELTY_ROUND_STARTED']}   "
          f"NOVELTY_PRODUCED {types['NOVELTY_PRODUCED']}")
    origins = collections.Counter()
    for e in ev:
        if e["type"] == "CANDIDATE_REGISTERED":
            origins[(e["payload"].get("candidate") or {}).get("origin")] += 1
    print(f"  candidate origins: {dict(origins)}")

    # --- 5. Freezes ---------------------------------------------------------------------
    # Family status is set on the family OBJECT (orchestrator.py:1661) and reaches the log
    # through the RUN_FINISHED families snapshot -- there is no `FAMILY_FROZEN` event. A first
    # version of this script looked for one and printed "none" for a run whose report showed two
    # frozen_converged families and one frozen_budget. `FAMILY_FROZEN_UNREWRITABLE` IS a real
    # event, for the narrower case of a family with nothing to rewrite.
    print("\n--- 5. FREEZES " + "-" * 62)
    shown = False
    if fin:
        s = fin["payload"].get("summary") or fin["payload"]
        fams = s.get("families") or {}
        if isinstance(fams, dict):
            for fid, f in fams.items():
                status = f.get("status") if isinstance(f, dict) else f
                rounds = f.get("rewrite_rounds_used") if isinstance(f, dict) else None
                hist = f.get("history") if isinstance(f, dict) else None
                warn = ""
                # The trap this exists to catch: `frozen_converged` on a family that never
                # actually evaluated a rewrite reads as "no structural headroom left" when the
                # truth is "we never looked".
                if status == "frozen_converged" and not rounds:
                    warn = "  <-- CONVERGED WITHOUT EVALUATING A REWRITE (headroom UNKNOWN)"
                print(f"  {fid}: {status}  rounds={rounds}  history={hist}{warn}")
                shown = True
    for e in ev:
        if e["type"] == "FAMILY_FROZEN_UNREWRITABLE":
            print(f"  {e['payload'].get('family_id')}: frozen_unrewritable "
                  f"({str(e['payload'].get('detail'))[:80]})")
            shown = True
    if not shown:
        print("  no freeze information (run unfinished, or no families frozen)")

    # --- 6. The fp16-ceiling defect ------------------------------------------------------
    print("\n--- 6. IMPOSSIBLE FRACTIONS (the missing fp16 ceiling) " + "-" * 23)
    imp = [e["payload"] for e in ev
           if e["type"] == "BOTTLENECK_CLASSIFIED"
           and (e["payload"].get("evidence") or {}).get("impossible_fraction")]
    print(f"  verdicts with impossible_fraction: {len(imp)} of {types['BOTTLENECK_CLASSIFIED']}")
    for p in imp[:6]:
        evd = p["evidence"]
        print(f"    {p.get('candidate_id')}: {evd.get('pct_of_compute_peak')}% of "
              f"{evd.get('compute_ceiling_used')}  (tensor cores: {evd.get('uses_tensor_cores')})")

    # --- 7. Median coverage (the fix from this session) ---------------------------------
    print("\n--- 7. MEDIAN COVERAGE " + "-" * 54)
    withm = without = 0
    for e in ev:
        if e["type"] == "TRIAL_DONE":
            lat = ((e["payload"].get("trial") or {}).get("latency_ms")) or {}
            if lat:
                withm += 1 if lat.get("median") is not None else 0
                without += 0 if lat.get("median") is not None else 1
    print(f"  trials with a median: {withm}   without: {without}")
    for e in ev:
        if e["type"] == "BASELINE_DONE":
            b = e["payload"].get("baseline") or {}
            lat = b.get("latency_ms") or {}
            print(f"    baseline {b.get('kind'):22s} mean={lat.get('mean')} "
                  f"median={lat.get('median')} samples={len(lat.get('samples') or [])}")

    # --- 8. Failure and rescue accounting ----------------------------------------------
    print("\n--- 8. FAILURES AND RESCUES " + "-" * 49)
    fk = collections.Counter()
    for e in ev:
        if e["type"] == "TRIAL_DONE":
            t = e["payload"].get("trial") or {}
            if t.get("status") != "complete":
                fk[t.get("failure_kind") or "?"] += 1
    print(f"  trial failures: {dict(fk)}")
    print(f"  AGENT_CALL_FAILED {types['AGENT_CALL_FAILED']}   "
          f"ARTIFACT_RESCUE {types['AGENT_ARTIFACT_RESCUE']}   "
          f"SESSION_RESET {types['AGENT_SESSION_RESET']}")
    print(f"  SPACE_REJECTED {types['SPACE_REJECTED']}   "
          f"SPACE_EXPANDED {types['SPACE_EXPANDED']}   "
          f"SPACE_EXPANSION_REJECTED {types['SPACE_EXPANSION_REJECTED']}")
    print(f"  KERNELS_NEVER_LAUNCHED {types['KERNELS_NEVER_LAUNCHED']}   "
          f"BOTTLENECK_CLASSIFY_FAILED {types['BOTTLENECK_CLASSIFY_FAILED']}")

    # --- 9. Space expansions: improved vs ATTRIBUTABLE ---------------------------------
    print("\n--- 9. SPACE EXPANSIONS: improved is not the same as attributable " + "-" * 12)
    spaces = collections.defaultdict(list)
    for e in ev:
        if e["type"] == "SPACE_PUBLISHED":
            sp = e["payload"].get("space") or {}
            if sp.get("candidate_id"):
                spaces[sp["candidate_id"]].append(
                    {d["name"]: list(d.get("choices") or []) for d in (sp.get("domains") or [])})
    for i, e in enumerate(ev):
        if e["type"] != "SPACE_EXPANDED":
            continue
        p = e["payload"]
        cid, prev = p.get("candidate_id"), p.get("prev_best_ms")
        after, win = None, None
        for e2 in ev[i:]:
            if e2["type"] != "TRIAL_DONE":
                continue
            t = e2["payload"].get("trial") or {}
            if t.get("candidate_id") != cid or t.get("status") != "complete":
                continue
            lat = t.get("latency_ms") or {}
            ms = lat.get("median") or lat.get("mean")
            if ms and (after is None or ms < after):
                after, win = ms, (t.get("params") or {}).get("values")
        gain = None if not (prev and after) else 100 * (prev - after) / prev
        print(f"  {cid}: {prev:.3f} -> {after:.3f} ms ({gain:+.1f}%)")
        vers = spaces.get(cid) or []
        if len(vers) >= 2 and win:
            added = {n: [v for v in vers[-1].get(n, []) if v not in vers[0].get(n, [])]
                     for n in vers[-1]}
            used_new = {n: win.get(n) for n, vals in added.items()
                        if vals and win.get(n) in vals}
            print(f"     newly added: { {k: v for k, v in added.items() if v} }")
            print(f"     winner uses a NEW value: {used_new or 'NO -- gain came from re-tuning'}")


if __name__ == "__main__":
    main(sys.argv[1])
