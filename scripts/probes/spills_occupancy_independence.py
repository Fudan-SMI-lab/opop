"""Before implementing an n_spills or occupancy wall: are those signals independent of what we have?

WHY THIS RUNS FIRST. The `logical_bytes` proposal was withdrawn after measurement showed it was the
tile product rescaled (rho +0.978) -- the third time in this project a new "dimension" turned out to
be an algebraic restatement of an existing one (pct_of_dram_peak and speed-of-light headroom were
both 1/latency at rho +1.000). The rule that came out of it applies here: a signal must be shown to
carry information the existing ones do not, BEFORE it is built. `n_spills` has a prior advantage --
it is the only dimension measured to have a stable resource-to-latency conversion rate -- but a prior
advantage is exactly what the other three had.

WHAT IS MEASURED, per (candidate, parameter set) row taken from finished runs' TRIAL_DONE profiles:

    n_spills     local-memory slots the compiler had to spill to
    occupancy    fraction of a SM's warp slots that are resident (the inverted dimension)
    n_regs       registers per thread
    shared_bytes shared memory per block
    tile_product product of the tile-shaped knobs in that parameter set
    latency      the trial's own robust_ms

Then Spearman between the two CANDIDATE signals and the four INCUMBENT ones. |rho| >= 0.9 against any
incumbent disqualifies the signal; that is the threshold the withdrawn proposal failed.

TWO THINGS THIS ALSO HAS TO ANSWER, because independence alone is not enough for a WALL:

  1. Does the signal VARY within a single candidate's own parameter sets? A signal that is constant
     per candidate cannot be truncated by any knob -- this is exactly how `logical_bytes` failed on
     elementwise kernels, where the total was identical for every BLOCK.
  2. Is the signal ever in its BINDING state at all? `n_spills` binds when non-zero and occupancy
     when below 0.30. A signal that never fires in the measured corpus has nothing to detect, and
     "we found no wall" would be indistinguishable from "the detector cannot fire" --
     the `an-unreachable-branch-is-not-a-safeguard` failure.

Reads profiles straight out of events.jsonl. `robust_ms` is a @property and is never serialized, so
it is recomputed here as median-else-mean rather than read.
"""
import json
import os
import re
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print(__doc__)
    print("usage: spills_occupancy_independence.py <run_dir> [<run_dir>...]")
    raise SystemExit(2)


def robust_ms(lat: dict) -> float | None:
    """Median if present, else mean. Mirrors LatencyStats.robust_ms, which is never serialized."""
    if not isinstance(lat, dict):
        return None
    for key in ("median", "mean"):
        v = lat.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def spearman(xs, ys):
    n = len(xs)
    if n < 4 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


TILE_RE = re.compile(r"BLOCK|TILE|^B[MNK]$|_B[MNK]$", re.I)
rows = []
for rd in sys.argv[1:]:
    path = os.path.join(rd, "events.jsonl")
    if not os.path.exists(path):
        print(f"MISSING {path}")
        continue
    label = os.path.basename(rd.rstrip("/\\"))
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
            vals = ((t.get("params") or {}).get("values")) or {}
            # occupancy is NESTED on the profile; a flat read turns "measured" into "unmeasured".
            occ_raw = prof.get("occupancy")
            occ = None
            if isinstance(occ_raw, dict):
                v = occ_raw.get("occupancy")
                occ = float(v) if isinstance(v, (int, float)) else None
            elif isinstance(occ_raw, (int, float)):
                occ = float(occ_raw)
            tile, n_tile = 1, 0
            for k, v in vals.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if TILE_RE.search(k):
                    tile *= int(v)
                    n_tile += 1
            rows.append({
                "run": label,
                "cand": t.get("candidate_id"),
                "n_spills": prof.get("n_spills"),
                "occupancy": occ,
                "occ_limiter": (occ_raw or {}).get("limiter") if isinstance(occ_raw, dict) else None,
                "n_regs": prof.get("n_regs"),
                "shared_bytes": prof.get("shared_bytes"),
                "tile_product": tile if n_tile else None,
                "ms": robust_ms(t.get("latency_ms") or {}),
            })

print(f"completed trials with a profile: {len(rows)}")
for field in ("n_spills", "occupancy", "n_regs", "shared_bytes", "tile_product", "ms"):
    n = sum(1 for r in rows if isinstance(r[field], (int, float)))
    print(f"  {field:14s} present on {n:5d} / {len(rows)}")
if not rows:
    raise SystemExit(0)

# --- Q2 first: can the detector ever fire? An unfireable detector makes Q1/Q3 moot. ------------
print()
print("=== Q2 CAN IT FIRE? the binding state has to occur in the corpus at all ===")
sp = [r["n_spills"] for r in rows if isinstance(r["n_spills"], (int, float))]
oc = [r["occupancy"] for r in rows if isinstance(r["occupancy"], (int, float))]
if sp:
    nz = sum(1 for v in sp if v > 0)
    print(f"  n_spills  > 0 (binding) on {nz:5d} / {len(sp)}  ({100.0 * nz / len(sp):.1f}%)"
          f"   max={max(sp)}")
    print("    => " + ("the detector CAN fire" if nz else
                       "**NEVER FIRES** -- nothing to detect in this corpus; a wall detector here "
                       "would be an unreachable branch"))
else:
    print("  n_spills  NOT MEASURED anywhere in this corpus")
if oc:
    low = sum(1 for v in oc if v < 0.30)
    near = sum(1 for v in oc if 0.30 <= v < 0.50)
    print(f"  occupancy < 0.30 (binding) on {low:5d} / {len(oc)}  ({100.0 * low / len(oc):.1f}%);"
          f"  0.30-0.50 near-binding on {near}")
    print(f"    range {min(oc):.3f} .. {max(oc):.3f}   median {statistics.median(oc):.3f}")
    print("    => " + ("the detector CAN fire" if low else
                       "**NEVER BINDS** -- occupancy never goes below 0.30 here"))
    lim = defaultdict(int)
    for r in rows:
        if r["occ_limiter"]:
            lim[str(r["occ_limiter"])] += 1
    if lim:
        print(f"    limiters seen: {dict(sorted(lim.items(), key=lambda kv: -kv[1]))}")
else:
    print("  occupancy NOT MEASURED anywhere in this corpus")

# --- Q1: does it vary WITHIN a candidate? -------------------------------------------------------
print()
print("=== Q1 WITHIN-CANDIDATE VARIATION: can a knob move it? ===")
print("    (a signal constant across a candidate's own parameter sets cannot be truncated by a knob")
print("     -- this is exactly how logical_bytes failed on elementwise kernels)")
for field in ("n_spills", "occupancy"):
    by_cand = defaultdict(list)
    for r in rows:
        if isinstance(r[field], (int, float)) and r["cand"]:
            by_cand[r["cand"]].append(float(r[field]))
    varying = flat = 0
    examples = []
    for cid, vs in by_cand.items():
        if len(vs) < 4:
            continue
        hi, lo = max(vs), min(vs)
        spread = (hi - lo) / hi if hi else 0.0
        if spread < 0.01:
            flat += 1
        else:
            varying += 1
            examples.append((spread, cid, lo, hi, len(vs)))
    total = varying + flat
    print(f"  {field:10s} candidates with >=4 rows: {total};  varies on {varying}, FLAT on {flat}")
    for spread, cid, lo, hi, n in sorted(examples, reverse=True)[:5]:
        print(f"      {cid} n={n:3d}  {lo:.4g} .. {hi:.4g}  spread {spread * 100:.1f}%")

# --- Q3: independence from the incumbents ------------------------------------------------------
print()
print("=== Q3 INDEPENDENCE: |rho| >= 0.9 against any incumbent DISQUALIFIES the signal ===")
print("    (the threshold logical_bytes failed: +0.978 against the tile product)")
for cand_field in ("n_spills", "occupancy"):
    print(f"  --- {cand_field}")
    for inc in ("n_regs", "shared_bytes", "tile_product", "ms"):
        xs, ys = [], []
        for r in rows:
            a, b = r[cand_field], r[inc]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                xs.append(float(a))
                ys.append(float(b))
        rho = spearman(xs, ys)
        if rho is None:
            print(f"      vs {inc:14s} rho None   (n={len(xs)}; a side is constant)")
            continue
        verdict = ("**DISQUALIFIED -- restatement**" if abs(rho) >= 0.9 else
                   "strongly related" if abs(rho) >= 0.6 else
                   "moderately related" if abs(rho) >= 0.3 else
                   "independent -- carries own information")
        print(f"      vs {inc:14s} rho {rho:+.3f}  n={len(xs)}  {verdict}")

# --- and the two candidate signals against EACH OTHER ------------------------------------------
xs = [float(r["n_spills"]) for r in rows
      if isinstance(r["n_spills"], (int, float)) and isinstance(r["occupancy"], (int, float))]
ys = [float(r["occupancy"]) for r in rows
      if isinstance(r["n_spills"], (int, float)) and isinstance(r["occupancy"], (int, float))]
rho = spearman(xs, ys)
print()
print(f"  n_spills vs occupancy  rho {'None' if rho is None else f'{rho:+.3f}'}  n={len(xs)}")
print("    (if these two restate each other, only ONE of them is worth building)")
