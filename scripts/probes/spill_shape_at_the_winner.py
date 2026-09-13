"""Does the same peaked-shape objection apply to n_spills, or is it monotone?

WHY THIS IS NEEDED FOR FAIRNESS. Occupancy failed its criterion test for a specific reason: the
occupancy-latency relation is PEAKED on 37 of 56 candidates and the winner sits at a median
normalised position 0.33 in its own range, so "raise occupancy" is advice against the measured
optimum. That objection is about the SHAPE of the relation, not about occupancy being a bad idea, so
it has to be put to n_spills too before n_spills is treated as the survivor -- otherwise the
conclusion would be "the signal I tested second wins", which is not a finding.

n_spills has a reason to behave differently: its binding state is a hard edge (zero) rather than a
threshold on a continuum, and spilling to local memory is slow in one direction only. If the relation
IS monotone -- more spills, slower -- then "reduce spills" is safe advice at the optimum in a way
"raise occupancy" is not. If it is peaked too, both soft walls have the same defect and neither
criterion can be written in the proposed form.

Measured per candidate: the shape of the spills-latency relation over the candidate's own trials, and
where the WINNER's spill count sits in its own range. Reported alongside how many winners spill at
all -- a winner with zero spills cannot be walled by spilling, and if most winners are at zero then
the detector, whatever its shape, has little left to find at the point that matters.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print(__doc__)
    print("usage: spill_shape_at_the_winner.py <run_dir> [<run_dir>...]")
    raise SystemExit(2)


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
            sp, ms = prof.get("n_spills"), robust_ms(t.get("latency_ms") or {})
            if not isinstance(sp, (int, float)) or ms is None:
                continue
            by_cand[t.get("candidate_id")].append({"sp": float(sp), "ms": ms,
                                                   "regs": prof.get("n_regs")})

shapes = defaultdict(int)
positions, winner_zero, winner_total = [], 0, 0
print("=== per candidate: where does the winner's SPILL COUNT sit in its own range? ===")
for cid, rs in sorted(by_cand.items()):
    if len(rs) < 8:
        continue
    sps = [r["sp"] for r in rs]
    lo, hi = min(sps), max(sps)
    win = min(rs, key=lambda r: r["ms"])
    winner_total += 1
    if win["sp"] == 0:
        winner_zero += 1
    pos = (win["sp"] - lo) / (hi - lo) if hi > lo else None
    if pos is not None:
        positions.append(pos)
    # Shape: terciles by spill count.
    s = sorted(rs, key=lambda r: r["sp"])
    k = max(1, len(s) // 3)
    m_lo = statistics.median(r["ms"] for r in s[:k])
    m_mid = statistics.median(r["ms"] for r in s[k:-k]) if len(s) > 2 * k else m_lo
    m_hi = statistics.median(r["ms"] for r in s[-k:])
    # For spills, "good" is LOW, so monotone-as-expected means latency rises with spills.
    if m_lo < m_mid < m_hi:
        shapes["monotone AS EXPECTED: more spills, slower"] += 1
    elif m_hi < m_mid < m_lo:
        shapes["monotone INVERTED: more spills, faster"] += 1
    elif m_mid <= m_lo and m_mid <= m_hi:
        shapes["PEAKED: the middle is fastest"] += 1
    else:
        shapes["other / non-monotone"] += 1
    print(f"  {cid}  spills {lo:.0f}..{hi:.0f}  winner {win['sp']:.0f}"
          f"  pos {'n/a' if pos is None else f'{pos:.2f}'}  regs={win['regs']}")

print()
print(f"candidates examined: {winner_total}")
print(f"  winners with ZERO spills: {winner_zero} / {winner_total}"
      f"  ({100.0 * winner_zero / winner_total:.0f}%)")
if positions:
    print(f"  winner's normalised position in its own spill range: "
          f"median {statistics.median(positions):.2f}  (0 = fewest spills, 1 = most)")
    at_bot = sum(1 for p in positions if p <= 0.2)
    print(f"  winners at the BOTTOM of their own spill range (<=0.2): {at_bot} / {len(positions)}")

print()
print("=== shape of the spills-latency relation ===")
for k, v in sorted(shapes.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
print()
print("READ THIS AGAINST OCCUPANCY: occupancy was PEAKED on 37 of 56 with the winner at a median")
print("position 0.33 of its own range, which is why 'raise occupancy' is advice against the measured")
print("optimum. If spills are monotone-as-expected AND winners sit at the bottom of their range,")
print("then 'reduce spills' is consistent with where the optimum actually is -- and only then is the")
print("proposed criterion writable for this signal.")
