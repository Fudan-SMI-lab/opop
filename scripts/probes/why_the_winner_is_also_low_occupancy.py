"""The contradiction the soft-wall probe surfaced, and it decides whether the criterion is usable.

TWO MEASURED FACTS THAT LOOK INCOMPATIBLE:
  * within a candidate, the high-occupancy third of its trials is a median 23.1% FASTER than the
    low-occupancy third (37 of 56 candidates, p90 +60.3%);
  * yet 55 of 56 candidates' OWN BEST trial sits below the 0.30 binding threshold.

If both hold, then "occupancy is binding" is true of the winner almost always, and a detector firing
on it would fire on essentially every candidate -- while the within-candidate trend says occupancy
still matters. Resolving this decides the criterion's form: a wall that fires on 55 of 56 winners is
a constant, not a finding.

THREE POSSIBILITIES, and they call for different criteria:
  A. The threshold is simply too high for this hardware/workload, and the winners cluster just under
     it. Then a RELATIVE criterion (this candidate's own achievable range) works where the absolute
     0.30 does not.
  B. The relationship is non-monotone: occupancy helps up to a point and the winner sits at an
     interior optimum. Then "raise occupancy" is wrong advice at the winner even though the trend is
     positive on average.
  C. The winner is limited by something else entirely and its low occupancy is incidental.

Measured here: where the winner's occupancy sits inside its candidate's own measured range, whether
the per-candidate occupancy-latency relation is monotone or peaked, and what the compiler names as
the limiter at the winner (`registers` vs `shared_memory` -- the two the corpus actually shows).
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print(__doc__)
    print("usage: why_the_winner_is_also_low_occupancy.py <run_dir> [<run_dir>...]")
    raise SystemExit(2)

OCC_BINDING_BELOW = 0.30


def robust_ms(lat):
    if not isinstance(lat, dict):
        return None
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


by_cand = defaultdict(list)
for rd in sys.argv[1:]:
    p = os.path.join(rd, "events.jsonl")
    if not os.path.exists(p):
        continue
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            t = (e.get("payload") or {}).get("trial") or {}
            if t.get("status") != "complete":
                continue
            prof = t.get("profile") or {}
            occ_raw = prof.get("occupancy") or {}
            occ = occ_raw.get("occupancy") if isinstance(occ_raw, dict) else occ_raw
            ms = robust_ms(t.get("latency_ms") or {})
            if not isinstance(occ, (int, float)) or ms is None:
                continue
            by_cand[t.get("candidate_id")].append({
                "occ": float(occ), "ms": ms,
                "limiter": (occ_raw.get("limiter") if isinstance(occ_raw, dict) else None),
                "regs": prof.get("n_regs"), "shared": prof.get("shared_bytes"),
                "spills": prof.get("n_spills")})

print("=== where does the WINNER sit inside its candidate's own occupancy range? ===")
positions, shapes, limiters = [], defaultdict(int), defaultdict(int)
for cid, rs in sorted(by_cand.items()):
    if len(rs) < 8:
        continue
    occs = [r["occ"] for r in rs]
    lo, hi = min(occs), max(occs)
    win = min(rs, key=lambda r: r["ms"])
    pos = (win["occ"] - lo) / (hi - lo) if hi > lo else None
    if pos is not None:
        positions.append(pos)
    limiters[str(win["limiter"])] += 1
    # Is latency monotone in occupancy, or peaked? Bucket into terciles by occupancy.
    s = sorted(rs, key=lambda r: r["occ"])
    k = max(1, len(s) // 3)
    m_lo = statistics.median(r["ms"] for r in s[:k])
    m_mid = statistics.median(r["ms"] for r in s[k:-k]) if len(s) > 2 * k else m_lo
    m_hi = statistics.median(r["ms"] for r in s[-k:])
    if m_hi < m_mid < m_lo:
        shapes["monotone: higher occupancy always faster"] += 1
    elif m_mid <= m_hi and m_mid <= m_lo:
        shapes["PEAKED: the middle is fastest"] += 1
    elif m_lo < m_mid < m_hi:
        shapes["monotone: LOWER occupancy faster"] += 1
    else:
        shapes["other / non-monotone"] += 1
    print(f"  {cid}  range {lo:.3f}..{hi:.3f}  winner {win['occ']:.3f}"
          f"  pos {'n/a' if pos is None else f'{pos:.2f}'}"
          f"  limiter={win['limiter']}  regs={win['regs']} shared={win['shared']}"
          f" spills={win['spills']}")

print()
if positions:
    print(f"winner's normalised position in its OWN occupancy range: "
          f"median {statistics.median(positions):.2f}  "
          f"(0 = candidate's lowest occupancy, 1 = its highest), n={len(positions)}")
    at_top = sum(1 for p in positions if p >= 0.8)
    at_bot = sum(1 for p in positions if p <= 0.2)
    print(f"  winners at the TOP of their own range (>=0.8): {at_top} / {len(positions)}")
    print(f"  winners at the BOTTOM of their own range (<=0.2): {at_bot} / {len(positions)}")
    print("  => " + (
        "the winner already sits at the HIGHEST occupancy this candidate can reach ⇒ the absolute "
        "0.30 threshold is what is wrong, not the signal: a RELATIVE criterion is required"
        if at_top > at_bot else
        "the winner is NOT at the top of its own range ⇒ occupancy is not what the winner traded on, "
        "and 'raise occupancy' would be advice against the measured optimum"))

print()
print("=== shape of the occupancy-latency relation, per candidate ===")
for k, v in sorted(shapes.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")

print()
print("=== what limits occupancy AT THE WINNER ===")
for k, v in sorted(limiters.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
print("  (this is the compiler's own attribution; it names which resource capped residency, so it")
print("   is the natural place a soft wall would point the rewriter)")
