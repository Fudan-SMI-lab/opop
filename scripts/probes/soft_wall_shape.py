"""Two follow-ups the independence probe raised, both of which decide whether a wall is buildable.

FOLLOW-UP 1 -- occupancy binds 87.6% of the time. A criterion that fires on seven trials out of
eight is not a detector, it is a constant. `three-same-cell-hits-are-not-a-model` and
`an-unreachable-branch-is-not-a-safeguard` are the two sides of this: a branch that never fires and
a branch that always fires are both uninformative. So the question is whether occupancy SEPARATES
anything within a candidate -- specifically, whether the trials with high occupancy are the fast ones.
If low occupancy is simply the normal state of every configuration including the winners, then
"occupancy is binding" cannot be evidence that a knob should be freed.

FOLLOW-UP 2 -- the wall shape. A wall needs the resource to TRUNCATE a knob's range, and for a soft
wall there is no refusal to read. The proposed criterion is "as knob K increases, the signal
monotonically worsens and reaches its binding state". This measures whether that pattern actually
occurs: per (candidate, knob), is the signal monotone in the knob's ordered values, and does it hit
the binding state at the top of the measured range while latency was still improving? That last
conjunct is what separates a wall worth reporting from one that costs nothing to leave alone -- the
same `tail_gain_pct > 0` filter the shared-memory wall already applies.

Both are read from the same finished-run profiles; nothing here needs a GPU.
"""
import json
import os
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print(__doc__)
    print("usage: soft_wall_shape.py <run_dir> [<run_dir>...]")
    raise SystemExit(2)

OCC_BINDING_BELOW = 0.30       # dimensions.py:75, the project's own threshold
SPILL_BINDING_ABOVE = 0        # any spill at all is binding (dimensions.py:265)


def robust_ms(lat):
    if not isinstance(lat, dict):
        return None
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


rows = []
for rd in sys.argv[1:]:
    path = os.path.join(rd, "events.jsonl")
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as fh:
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
            occ_raw = prof.get("occupancy")
            occ = None
            if isinstance(occ_raw, dict):
                v = occ_raw.get("occupancy")
                occ = float(v) if isinstance(v, (int, float)) else None
            ms = robust_ms(t.get("latency_ms") or {})
            if ms is None:
                continue
            rows.append({"cand": t.get("candidate_id"),
                         "vals": ((t.get("params") or {}).get("values")) or {},
                         "occ": occ,
                         "spills": prof.get("n_spills"),
                         "ms": ms})

print(f"rows={len(rows)}")

# ---------------------------------------------------------------- follow-up 1
print()
print("=== 1. does occupancy SEPARATE the fast trials from the slow ones, within a candidate? ===")
print("    (it binds on 87.6% of trials; if the winners are also low-occupancy then 'binding'")
print("     cannot be evidence that a knob needs freeing)")
by_cand = defaultdict(list)
for r in rows:
    if r["occ"] is not None and r["cand"]:
        by_cand[r["cand"]].append(r)

better_when_high = worse_when_high = no_signal = 0
winner_binding = winner_total = 0
deltas = []
for cid, rs in by_cand.items():
    if len(rs) < 8:
        continue
    rs_sorted = sorted(rs, key=lambda r: r["ms"])
    # The candidate's own best trial: is IT low-occupancy too?
    winner_total += 1
    if rs_sorted[0]["occ"] < OCC_BINDING_BELOW:
        winner_binding += 1
    # Median latency of the top-occupancy third vs the bottom-occupancy third.
    by_occ = sorted(rs, key=lambda r: r["occ"])
    k = max(1, len(by_occ) // 3)
    lo_occ = statistics.median(r["ms"] for r in by_occ[:k])
    hi_occ = statistics.median(r["ms"] for r in by_occ[-k:])
    d = (lo_occ - hi_occ) / lo_occ * 100.0 if lo_occ else 0.0
    deltas.append(d)
    if d > 5:
        better_when_high += 1
    elif d < -5:
        worse_when_high += 1
    else:
        no_signal += 1

print(f"  candidates with >=8 rows: {better_when_high + worse_when_high + no_signal}")
print(f"    high occupancy is FASTER  (>5%): {better_when_high}")
print(f"    high occupancy is SLOWER  (>5%): {worse_when_high}")
print(f"    no separation (within +-5%)     : {no_signal}")
if deltas:
    print(f"    median separation: {statistics.median(deltas):+.1f}%  "
          f"(p10 {sorted(deltas)[len(deltas) // 10]:+.1f}%, "
          f"p90 {sorted(deltas)[9 * len(deltas) // 10]:+.1f}%)")
print(f"  the candidate's OWN best trial is itself below 0.30: "
      f"{winner_binding} / {winner_total}")
print("  => " + ("occupancy does NOT separate winners from losers here, so 'occupancy is binding' "
                 "is the normal state and cannot by itself justify freeing a knob"
                 if better_when_high <= max(1, (better_when_high + worse_when_high + no_signal) // 4)
                 else "occupancy DOES separate: higher occupancy is measurably faster, so the "
                      "binding verdict carries information"))

# ---------------------------------------------------------------- follow-up 2
print()
print("=== 2. WALL SHAPE: does 'knob up => signal worsens monotonically into the binding state'")
print("       actually occur, with latency still improving toward that end? ===")


def as_num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


found = {"occupancy": [], "n_spills": []}
checked = 0
for cid, rs in sorted(by_cand.items()):
    if len(rs) < 8:
        continue
    knobs = set()
    for r in rs:
        knobs.update(r["vals"].keys())
    for knob in sorted(knobs):
        # Per ordered value: median signal and median latency.
        buckets = defaultdict(list)
        for r in rs:
            x = as_num(r["vals"].get(knob))
            if x is not None:
                buckets[x].append(r)
        if len(buckets) < 3:
            continue
        checked += 1
        xs = sorted(buckets)
        for field, binding in (("occupancy", lambda v: v < OCC_BINDING_BELOW),
                               ("n_spills", lambda v: v > SPILL_BINDING_ABOVE)):
            key = "occ" if field == "occupancy" else "spills"
            sig, lat = [], []
            ok = True
            for x in xs:
                vs = [r[key] for r in buckets[x] if isinstance(r[key], (int, float))]
                if not vs:
                    ok = False
                    break
                sig.append(statistics.median(vs))
                lat.append(statistics.median(r["ms"] for r in buckets[x]))
            if not ok or len(sig) < 3:
                continue
            # Worsening means DOWN for occupancy, UP for spills.
            if field == "occupancy":
                worsens = all(sig[i + 1] <= sig[i] for i in range(len(sig) - 1)) and sig[-1] < sig[0]
            else:
                worsens = all(sig[i + 1] >= sig[i] for i in range(len(sig) - 1)) and sig[-1] > sig[0]
            if not worsens or not binding(sig[-1]):
                continue
            # Latency still improving toward the walled end -- the tail_gain_pct > 0 filter.
            tail = lat[-3:]
            gain = (tail[0] - tail[-1]) / tail[0] * 100.0 if tail[0] else 0.0
            found[field].append({"cand": cid, "knob": knob, "n_values": len(xs),
                                 "sig_first": sig[0], "sig_last": sig[-1],
                                 "tail_gain_pct": gain,
                                 "worth": gain > 0})

print(f"  (candidate, knob) pairs examined: {checked}")
for field, hits in found.items():
    worth = [h for h in hits if h["worth"]]
    print(f"  {field:10s} monotone-into-binding: {len(hits):3d};  "
          f"of those, latency still improving: {len(worth):3d}")
    for h in sorted(worth, key=lambda h: -h["tail_gain_pct"])[:8]:
        print(f"      {h['cand']} {h['knob']:22s} {h['n_values']} values  "
              f"{h['sig_first']:.4g} -> {h['sig_last']:.4g}  tail gain {h['tail_gain_pct']:+.1f}%")
print("  => a soft-wall detector would report the 'latency still improving' rows; the difference")
print("     between the two columns is what the slope filter removes as not worth a rewrite.")
