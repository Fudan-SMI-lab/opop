"""N1: the SEED-PAIRED null distribution — what end-to-end difference does pure noise make?

WHY THIS IS THE DENOMINATOR FOR EVERYTHING END-TO-END. Two arms with byte-identical configs
and the SAME frozen candidate set still finish at different latencies. That difference is
the floor below which no P3 claim can be read. The historical ±2.35% was UNPAIRED (different
generator output per arm), so it conflated seed luck with noise; this pair removes seed luck
by construction and prices what is left.

WHAT IT MUST NOT DO. Report one number. A single "arms differ by X%" hides the three things
that decide whether X is noise:

  * WHICH candidate won in each arm. Both arms search the same four seeds, so if they end on
    the same candidate the difference is measurement noise plus search-path noise; if they
    end on DIFFERENT candidates it also contains "which lineage got lucky", which is a
    different and larger source (the recorded step-4 lesson: an unpaired difference of
    -2.31% came entirely from a candidate only one arm had).
  * PER-CANDIDATE agreement. The same seed, tuned twice under identical settings, gives two
    tuned latencies. The spread of THOSE is the cleanest noise estimate available, and it is
    per-candidate rather than per-arm, so it does not depend on which lineage won.
  * TOTAL GPU WALL CLOCK, not trial count. Measured on this very pair: 680 vs 741 trials,
    entirely explained by one arm expanding its space three times and the other once. Equal
    per-space budget is not equal total search, and a trial-count comparison would call a
    healthy A/A pair asymmetric.

`final_reeval_ms` is the number to compare, not `best_ms`: the tuned best is selected as a
minimum over trials and is therefore biased low by selection, while the re-eval is an
independent measurement of the chosen configuration. Its SIGN is not stable (recorded:
+-2-4% between a tuned best and its re-eval), which is exactly why the pair is needed.

Usage:  python v41_n1_noise_floor.py <arm_a_run_dir> <arm_b_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: Path) -> list[dict]:
    out = []
    for line in (run / "events.jsonl").open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def arm(run: Path) -> dict:
    ev = load(run)
    fin = [e for e in ev if e.get("type") == "RUN_FINISHED"]
    span = ((ev[-1].get("ts", 0) - ev[0].get("ts", 0)) / 3600.0) if len(ev) > 1 else 0.0

    # seed sha -> the candidate id this arm gave it, so the arms can be matched on the
    # SOURCE rather than on the id (ids are generated per run; matching on them reads a
    # correctly paired run as having zero shared candidates).
    sha_of: dict[str, str] = {}
    origin_of: dict[str, str] = {}
    for e in ev:
        if e.get("type") != "CANDIDATE_REGISTERED":
            continue
        c = (e.get("payload") or {}).get("candidate") or {}
        cid = str(c.get("candidate_id"))
        origin_of[cid] = str(c.get("origin"))
        if c.get("origin") == "seed" and c.get("source_sha"):
            sha_of[str(c["source_sha"])] = cid

    # Per-candidate tuned best, from TUNING_DONE (the last one per candidate: a space
    # expansion re-tunes the same candidate and the later value supersedes).
    tuned: dict[str, float] = {}
    for e in ev:
        if e.get("type") != "TUNING_DONE":
            continue
        p = e.get("payload") or {}
        cid = str(p.get("candidate_id"))
        v = p.get("best_ms")
        if v is None:
            v = ((p.get("best") or {}).get("latency_ms") or {}).get("median")
        if v is not None:
            tuned[cid] = float(v)

    trials = sum(1 for e in ev if e.get("type") == "TRIAL_DONE")
    expansions = sum(1 for e in ev if e.get("type") == "SPACE_EXPANDED")
    # Total GPU wall clock: the sum of measured job walls where journalled, else None so the
    # caller says "unavailable" instead of comparing trial counts by accident.
    job_wall = 0.0
    seen_wall = False
    for e in ev:
        p = e.get("payload") or {}
        w = p.get("job_wall_s")
        if w is None:
            w = ((p.get("trial") or {}).get("job_wall_s"))
        if w is not None:
            try:
                job_wall += float(w)
                seen_wall = True
            except (TypeError, ValueError):
                pass

    summary = ((fin[-1].get("payload") or {}).get("summary") or {}) if fin else {}
    best = summary.get("best") or {}
    return {
        "run": run.name, "arm": run.parent.name, "finished": bool(fin), "span_h": span,
        "trials": trials, "expansions": expansions,
        "gpu_wall_h": (job_wall / 3600.0) if seen_wall else None,
        "tuned": tuned, "sha_of": sha_of, "origin_of": origin_of,
        "best_cid": str(best.get("candidate_id")) if best else None,
        "best_ms": best.get("tuned_ms") or best.get("best_ms"),
        "reeval_ms": best.get("final_reeval_ms"),
        "summary": summary,
    }


def pct(a: float, b: float) -> float:
    """b relative to a, in percent. Positive = b is SLOWER."""
    return (b - a) / a * 100.0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    A, B = arm(Path(sys.argv[1].rstrip("/"))), arm(Path(sys.argv[2].rstrip("/")))

    print("=" * 74)
    print("N1 seed-paired A/A: %s  vs  %s" % (A["arm"], B["arm"]))
    print("=" * 74)
    for x in (A, B):
        print("  %-8s %s  finished=%s  span=%.2fh  trials=%d  expansions=%d  gpu_wall=%s"
              % (x["arm"], x["run"], x["finished"], x["span_h"], x["trials"],
                 x["expansions"],
                 ("%.2fh" % x["gpu_wall_h"]) if x["gpu_wall_h"] is not None else "n/a"))
    if not (A["finished"] and B["finished"]):
        print("\n!! at least one arm has not finished — the end-to-end difference is NOT")
        print("!! readable yet (a run's best improves monotonically until it stops).")
        return 2

    # 1. Seed pairing, on the SOURCE SHA. This is the precondition for everything else.
    shared = sorted(set(A["sha_of"]) & set(B["sha_of"]))
    print("\n-- seed pairing (matched on source_sha, never on candidate_id) --")
    print("  seeds in A: %d   in B: %d   SHARED: %d"
          % (len(A["sha_of"]), len(B["sha_of"]), len(shared)))
    if len(shared) != len(A["sha_of"]) or len(shared) != len(B["sha_of"]):
        print("  !! the arms did NOT search the same seed set — this is not an A/A pair.")
        print("  !! Every difference below also contains seed luck; do not use it as a floor.")

    # 2. Per-candidate agreement: the cleanest noise estimate, independent of which won.
    print("\n-- per-seed tuned best, same seed tuned twice (the cleanest noise read) --")
    diffs = []
    for sha in shared:
        ca, cb = A["sha_of"][sha], B["sha_of"][sha]
        ta, tb = A["tuned"].get(ca), B["tuned"].get(cb)
        if ta is None or tb is None:
            print("  %s  A=%-9s %-9s   B=%-9s %-9s   (one arm never finished tuning it)"
                  % (sha[:12], ca[:9], "n/a" if ta is None else "%.4f" % ta,
                     cb[:9], "n/a" if tb is None else "%.4f" % tb))
            continue
        d = pct(ta, tb)
        diffs.append(d)
        print("  %s  A=%.4f ms   B=%.4f ms   B-A = %+.2f%%" % (sha[:12], ta, tb, d))
    if diffs:
        ad = sorted(abs(d) for d in diffs)
        med = ad[len(ad) // 2] if len(ad) % 2 else (ad[len(ad) // 2 - 1] + ad[len(ad) // 2]) / 2
        print("  |B-A| over %d paired seeds: min %.2f%%  median %.2f%%  max %.2f%%"
              % (len(ad), ad[0], med, ad[-1]))
        print("  => THE PAIRED NOISE FLOOR for a per-candidate claim is the max: %.2f%%" % ad[-1])

    # 3. The end-to-end difference, and whether it is even the same lineage.
    print("\n-- end-to-end (the P3 denominator) --")
    for x in (A, B):
        print("  %-8s winner %s (origin %s)  tuned %s  re-eval %s"
              % (x["arm"], x["best_cid"], x["origin_of"].get(x["best_cid"], "?"),
                 x["best_ms"], x["reeval_ms"]))
    if A["reeval_ms"] and B["reeval_ms"]:
        d = pct(float(A["reeval_ms"]), float(B["reeval_ms"]))
        print("  final_reeval_ms difference (B vs A): %+.2f%%" % d)
        same_sha = None
        # Did both arms end on the same SEED lineage? Rewrite winners have no shared sha, so
        # this is reported as unknown rather than guessed.
        if (A["origin_of"].get(A["best_cid"]) == "seed"
                and B["origin_of"].get(B["best_cid"]) == "seed"):
            ra = {v: k for k, v in A["sha_of"].items()}.get(A["best_cid"])
            rb = {v: k for k, v in B["sha_of"].items()}.get(B["best_cid"])
            same_sha = (ra is not None and ra == rb)
        print("  same winning lineage: %s"
              % ("yes (same seed sha)" if same_sha else
                 "no / not comparable (at least one winner is a rewrite)"
                 if same_sha is False else "unknown (winner origin is not a seed)"))
        if diffs:
            print("\n  READING: |%.2f%%| end-to-end against a per-candidate floor of %.2f%%."
                  % (abs(d), max(abs(x) for x in diffs)))
            if abs(d) <= max(abs(x) for x in diffs):
                print("  The end-to-end gap is INSIDE the per-candidate noise ⇒ P3 in the main")
                print("  pair can only be read as descriptive; a difference of this size proves")
                print("  nothing either way.")
            else:
                print("  The end-to-end gap EXCEEDS the per-candidate floor ⇒ the extra comes from")
                print("  search-path divergence (which lineage won), NOT from measurement noise.")
                print("  P3 must be priced against THIS number, not against the per-candidate one.")
    else:
        print("  re-eval missing on at least one arm — report best_ms and say so.")

    # 4. Comparability by total GPU wall clock, not trials.
    print("\n-- comparability (total GPU wall clock, NOT trial count) --")
    if A["gpu_wall_h"] is not None and B["gpu_wall_h"] is not None:
        print("  A %.2fh vs B %.2fh  (B-A = %+.1f%%)"
              % (A["gpu_wall_h"], B["gpu_wall_h"], pct(A["gpu_wall_h"], B["gpu_wall_h"])))
    else:
        print("  job_wall_s not journalled in these runs — comparability cannot be settled")
        print("  from the event log; do NOT substitute trial counts (%d vs %d here, a gap"
              % (A["trials"], B["trials"]))
        print("  fully explained by %d vs %d space expansions)."
              % (A["expansions"], B["expansions"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
