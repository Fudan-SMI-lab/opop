"""Did the K expansion's win come from smaller tiles, and does shared memory explain it?

Box 3's best went 12.2844 -> 3.6055 ms (3.41x) when the K expansion added BL=16, BN=32, BP=16 and
the winner used all three. That is unusually clean attribution: the winning values did not exist in
the previous space, and `SPACE_EXPANDED` names exactly those three knobs with `direction: "min"`.

The interesting part is WHY, and the obvious answer is wrong. The winner carries 298 register spills
and sits at the 255-register cap -- so "fewer spills is faster" does not explain it. Every completed
trial on this candidate is at or near 255 regs, which means on this kernel the register file is
saturated regardless of configuration and spills are a CONSEQUENCE of tile size, not an independent
knob. So this ranks the candidate explanations by measurement instead of asserting one:

  * shared_bytes  -- the winner uses 12288 against the runner-up's 114944 (9.4x less)
  * n_spills      -- the winner has the fewest, but so do other slow configs
  * tensor_core   -- SASS tensor-core instruction count, from the disassembly
  * occupancy proxy: shared_bytes per block against the A800's 166912 opt-in limit

Prints Spearman rank correlation (not Pearson: latency spans an order of magnitude and the
relationship need not be linear) with n beside every coefficient, because a correlation over 22
points is a hint and this project has a recorded rule about reporting a rate without its denominator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The A800's measured per-block opt-in shared-memory limit.
SHARED_OPTIN = 166912


def rows(rd: Path) -> list[dict]:
    out = []
    with (rd / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            if tr.get("failure_kind"):
                continue
            m = (tr.get("latency_ms") or {}).get("median")
            pr = tr.get("profile") or {}
            if not isinstance(m, (int, float)) or not pr:
                continue
            v = (tr.get("params") or {}).get("values") or {}
            sass = pr.get("sass") or {}
            out.append({
                "ms": float(m),
                "regs": pr.get("n_regs"),
                "spills": pr.get("n_spills"),
                "shared": pr.get("shared_bytes"),
                "tc": sass.get("tensor_core"),
                "warps": pr.get("num_warps"),
                "stages": pr.get("num_stages"),
                "space": tr.get("space_id"),
                "params": v,
                # Tile volume: the product of whatever tile knobs this space has. Named generically
                # because the knob NAMES differ per candidate (BL/BN/BP here, BLOCK_M/BLOCK_N there).
                "tile": _tile_volume(v),
                "dtype": v.get("COMPUTE_DTYPE"),
            })
    return out


def _tile_volume(v: dict) -> int | None:
    """Product of the integer tile knobs, whatever they are called on this candidate.

    Hardcoding BL/BN/BP would make this script candidate-specific, which is exactly the kind of
    per-case special-casing this project forbids. The rule is structural: an integer knob whose name
    is not one of the known non-tile knobs.
    """
    NON_TILE = {"NUM_WARPS", "NUM_STAGES", "GROUP_M", "GEMM_GROUP_M"}
    vol = 1
    seen = 0
    for k, val in v.items():
        if k in NON_TILE or not isinstance(val, int) or isinstance(val, bool):
            continue
        vol *= val
        seen += 1
    return vol if seen else None


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. Ties get average ranks, which matters here: every trial is at 255 regs, so
    a tie-blind implementation would report a spurious coefficient on a constant column."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None       # a constant column has no rank correlation, it is not 0.0
    return num / (dx * dy)


def _populations(rs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split by register occupancy class before correlating anything.

    THIS IS NOT COSMETIC. On box 3's 22 trials the register count is bimodal -- 32 or 128-255, with
    nothing between -- and every trial slower than 100 ms is in the 32-register group. Pooling them
    produces Simpson's paradox in both directions at once:

        pooled   shared_bytes rho = +0.176   ("shared memory does not matter")
        within A shared_bytes rho = +0.795   (n=9)
        within B shared_bytes rho = +0.676   (n=13)

    The pooled coefficient HID a strong effect present in both groups. And the pooled n_spills
    rho = +0.924 is inflated by the same split, because the 32-register group has both the most
    spills and the worst latency for a reason that is not the spilling. Reporting either pooled
    number alone would have been a finding in the wrong direction -- the same shape as the recorded
    cross-dimension argmax error, where a normalization difference decided the answer.
    """
    hi = [r for r in rs if isinstance(r["regs"], int) and r["regs"] >= 128]
    lo = [r for r in rs if isinstance(r["regs"], int) and r["regs"] < 128]
    out = []
    if hi:
        out.append(("regs>=128", hi))
    if lo:
        out.append(("regs<128", lo))
    if len(out) < 2:
        return [("all", rs)]
    return out + [("POOLED (see docstring)", rs)]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rs = rows(Path(argv[0]))
    if not rs:
        raise SystemExit("no completed trials with a profile in %s" % argv[0])
    rs.sort(key=lambda r: r["ms"])

    print("completed trials with a profile: %d" % len(rs))
    print("\n%9s %5s %6s %8s %6s %6s %6s  tile      config" % (
        "med_ms", "regs", "spill", "shared", "tc", "warps", "stg"))
    for r in rs[:10]:
        print("%9.4f %5s %6s %8s %6s %6s %6s  %-9s %s" % (
            r["ms"], r["regs"], r["spills"], r["shared"], r["tc"], r["warps"], r["stages"],
            r["tile"], r["dtype"]))

    print("\n--- Spearman rank correlation with median latency, WITHIN register class ---")
    print("    positive = larger value goes with SLOWER; a constant column reports NO CORRELATION.")
    print("    Pooling register classes inverts these: see the module docstring for the measured")
    print("    Simpson's paradox (shared_bytes reads +0.18 pooled, +0.80/+0.68 within).")
    for pop_name, grp in _populations(rs):
        ms = [r["ms"] for r in grp]
        print("\n    %s  (n=%d, latency %.2f-%.2f ms)" % (
            pop_name, len(grp), min(ms), max(ms)))
        for key, label in (("shared", "shared_bytes"), ("spills", "n_spills"),
                           ("regs", "n_regs"), ("tc", "sass tensor_core"),
                           ("tile", "tile volume"), ("warps", "num_warps"),
                           ("stages", "num_stages")):
            vals = [r[key] for r in grp]
            if any(v is None for v in vals):
                print("      %-18s absent on %d of %d -- not comparable" % (
                    label, sum(1 for v in vals if v is None), len(vals)))
                continue
            rho = spearman([float(v) for v in vals], ms)
            if rho is None:
                uniq = sorted(set(vals))
                print("      %-18s NO CORRELATION: constant at %s across all %d" % (
                    label, uniq[0] if len(uniq) == 1 else uniq, len(vals)))
                continue
            # n<8 makes a rank correlation a hint, not a result. Say so beside the number rather
            # than in a footnote nobody reads.
            note = "   (n=%d: a hint, not a result)" % len(grp) if len(grp) < 8 else ""
            print("      %-18s rho = %+.3f%s" % (label, rho, note))

    # The winner against the runner-up, which is the comparison the expansion actually made.
    best, second = rs[0], next((r for r in rs[1:] if r["ms"] > rs[0]["ms"]), None)
    if second:
        print("\n--- winner vs runner-up ---")
        print("    latency  %8.4f -> %8.4f ms   (%.2fx)" % (
            second["ms"], best["ms"], second["ms"] / best["ms"]))
        for key, label in (("shared", "shared_bytes"), ("spills", "n_spills"),
                           ("tc", "tensor_core"), ("tile", "tile volume")):
            a, b = second.get(key), best.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
                print("    %-13s %8s -> %8s        (%.2fx)" % (label, a, b, a / b))
        if isinstance(best["shared"], (int, float)):
            print("    shared as a fraction of the A800 opt-in limit (%d): winner %.1f%%, "
                  "runner-up %.1f%%" % (SHARED_OPTIN, 100.0 * best["shared"] / SHARED_OPTIN,
                                        100.0 * second["shared"] / SHARED_OPTIN))
        print("\n    Blocks resident per SM is capped by shared memory: %d bytes allows %d blocks, "
              "%d allows %d. That is the mechanism a smaller tile buys, and it is measured here "
              "rather than inferred from occupancy (which Triton reports against a register cap "
              "this kernel hits regardless)." % (
                  best["shared"], SHARED_OPTIN // max(1, best["shared"] or 1),
                  second["shared"], SHARED_OPTIN // max(1, second["shared"] or 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
