"""Audit: how many correctness rejections are of candidates MORE consistent than the reference?

Each L3 task has a NOISE FLOOR: the reference model computed two ways (ieee fp32 vs tf32) does
not agree with itself perfectly. The failure detail records that floor next to the candidate's
own agreement, so every rejection can be classified by which is larger.

A candidate ABOVE the floor matched the reference better than the reference matched itself, and
was rejected anyway. A candidate BELOW it is genuinely less consistent.

Prints both groups with margins, because the decision this informs (a floor-relative gate,
docs/decisions-awaiting-user.md item 1, deferred twice by the user) turns on the margins and not
on the headline count: the below-floor group clusters within ~0.002 of the floor, so a
floor-relative rule has to pick a tolerance that also decides those.

See docs/measurement-rejections-above-the-noise-floor.md

Usage:  python scripts/audit_noise_floor_rejections.py [runs/run-... ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FRAC = re.compile(r"'frac_within_tol': ([0-9.]+)")
DTYPE = re.compile(r"'(?:COMPUTE_DTYPE|COMPUTE_PRECISION|DOT_PRECISION)': '(\w+)'")

# Metrics the detail prints for the candidate and for the reference-vs-itself floor.
# frac_within_tol and cosine are better when LARGER; the error metrics when SMALLER.
LARGER_IS_BETTER = ("frac_within_tol", "cosine")
METRICS = ("frac_within_tol", "median_rel_err", "max_abs_diff", "p99_rel_err")


def block(detail: str, marker: str) -> dict[str, float]:
    """Parse the metric dict that follows `marker` in a failure detail."""
    i = detail.find(marker)
    if i < 0:
        return {}
    seg = detail[i:i + 420]
    out: dict[str, float] = {}
    for key in ("frac_within_tol", "cosine", "median_rel_err", "p99_rel_err", "max_abs_diff"):
        m = re.search(rf"'{key}': '?([0-9.eE+-]+)'?", seg)
        if m:
            out[key] = float(m.group(1))
    return out


def four_metric_verdict(detail: str) -> dict[str, bool]:
    """Per-metric: is the candidate at least as good as the reference-vs-itself?

    The gate accepts on EITHER reference, so the candidate's figure per metric is the better
    of its ieee and tf32 comparisons. Returns {} when the detail cannot be parsed.
    """
    ieee, tf32, floor = (block(detail, "vs ieee ref"), block(detail, "vs tf32 ref"),
                         block(detail, "noise floor"))
    if not ieee or not floor:
        return {}
    verdict: dict[str, bool] = {}
    for key in METRICS:
        if key not in ieee or key not in floor:
            continue
        a, b = ieee[key], tf32.get(key)
        best = (max(a, b) if b is not None else a) if key in LARGER_IS_BETTER else (
            min(a, b) if b is not None else a)
        verdict[key] = best > floor[key] if key in LARGER_IS_BETTER else best < floor[key]
    return verdict


def parse(detail: str) -> tuple[float, float] | None:
    """(best candidate frac across the two references, task floor) or None.

    The detail prints three frac_within_tol values in a fixed order: vs ieee ref, vs tf32 ref,
    then the reference's own ieee-vs-tf32 spread. The gate accepts on EITHER reference, so the
    candidate's figure is the max of the first two.
    """
    found = FRAC.findall(detail or "")
    if len(found) < 3:
        return None
    return max(float(found[0]), float(found[1])), float(found[2])


def main(argv: list[str]) -> int:
    runs = [Path(a) for a in argv] or sorted(REPO.glob("runs/run-l3-*"))

    space_rows: list[tuple] = []
    trial_above = trial_below = 0

    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        task = "-".join(run.name.split("-")[1:3])
        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED"):
                pl = e["payload"]
                if "mismatch" not in (pl.get("detail") or ""):
                    continue
                parsed = parse(pl["detail"])
                if not parsed:
                    continue
                cand, floor = parsed
                space_rows.append((task, pl.get("candidate_id"), pl.get("reason"),
                                   cand, floor, cand > floor,
                                   four_metric_verdict(pl["detail"])))
            elif e["type"] == "TRIAL_DONE":
                t = e["payload"]["trial"]
                if t.get("failure_kind") != "correctness_mismatch":
                    continue
                parsed = parse(t.get("failure_detail") or "")
                if not parsed:
                    continue
                cand, floor = parsed
                if cand > floor:
                    trial_above += 1
                else:
                    trial_below += 1

    above = [r for r in space_rows if r[5]]
    below = [r for r in space_rows if not r[5]]

    print("=" * 82)
    print("SPACE/EXPANSION rejections for a correctness mismatch")
    print("=" * 82)
    n = len(space_rows)
    if not n:
        print("  none found")
        return 0
    print(f"  candidate ABOVE the floor (more consistent than the reference-vs-itself): "
          f"{len(above)}  ({100 * len(above) / n:.0f}%)")
    print(f"  candidate BELOW the floor (genuinely less consistent):                    "
          f"{len(below)}")

    print("\n  ABOVE -- rejected although more consistent than the reference is with itself.")
    print("  The `metrics` column is the FOUR-metric check: frac_within_tol alone (the metric the")
    print("  gate uses) can clear the floor while median/max deviation are multiples worse, so a")
    print("  4/4 row is verified clean and anything less is not.")
    for task, cid, reason, cand, floor, _, v in sorted(above, key=lambda r: -(r[3] - r[4])):
        n_ok, n_tot = sum(v.values()), len(v)
        tag = "CLEAN" if v and n_ok == n_tot else "gate-only -- DEGRADED on another metric"
        print(f"    {task:<8} {str(cid):<16} {str(reason):<26} "
              f"cand={cand:.6f} floor={floor:.6f}  +{cand - floor:.4f}  "
              f"{n_ok}/{n_tot}  {tag}")

    print("\n  BELOW -- and note how thin the margins are; a floor-relative gate must pick a")
    print("  tolerance that decides these too:")
    for task, cid, reason, cand, floor, _, v in sorted(below, key=lambda r: -(r[3] - r[4])):
        n_ok, n_tot = sum(v.values()), len(v)
        print(f"    {task:<8} {str(cid):<16} {str(reason):<26} "
              f"cand={cand:.6f} floor={floor:.6f}  {cand - floor:+.4f}  {n_ok}/{n_tot}")

    clean = [r for r in above if r[6] and sum(r[6].values()) == len(r[6])]
    gate_only = [r for r in above if r[6] and sum(r[6].values()) < len(r[6])]
    print(f"\n  FOUR-METRIC SUMMARY: {len(clean)} verified clean on every metric, "
          f"{len(gate_only)} above the floor on frac_within_tol but degraded elsewhere, "
          f"{len(below)} below.")
    print("  A floor-relative rule keyed on frac_within_tol ALONE would accept the degraded rows;")
    print("  one that also requires matching the reference on median and max deviation would not.")

    both = {r[1] for r in above} & {r[1] for r in below}
    if both:
        print(f"\n  Candidates appearing on BOTH sides (one witness above, another below): "
              f"{', '.join(sorted(str(c) for c in both))}")
        print("  Any floor-relative rule needs an answer for these; 'accept if either witness")
        print("  clears the floor' is the relaxation that makes the guarantee hardest to state.")

    if trial_above or trial_below:
        print()
        print("=" * 82)
        print("TRIAL-level mismatches (same comparison, for scale only)")
        print("=" * 82)
        print(f"  above the floor: {trial_above}    below: {trial_below}")
        print("  Do NOT read the above count as distinct losses -- it is dominated by a few")
        print("  candidates re-sampling the same configuration across a 40-trial budget.")

    # What does a floor-passing rejection make the repair loop DO? Group each candidate's
    # successive rejections and surface the ones where the compute dtype changed between
    # attempts -- that is repair reaching for the precision lever on a false signal.
    print()
    print("=" * 82)
    print("REPAIR CONSEQUENCE -- rejections where the dtype changed between attempts")
    print("=" * 82)
    found_any = False
    for run in runs:
        ev_path = run / "events.jsonl"
        if not ev_path.exists():
            continue
        seqs: dict[str, list[tuple]] = {}
        for line in ev_path.open(encoding="utf-8"):
            e = json.loads(line)
            if e["type"] not in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED"):
                continue
            pl = e["payload"]
            cid = pl.get("candidate_id")
            if not cid:
                continue
            detail = pl.get("detail") or ""
            dt = DTYPE.search(detail)
            parsed = parse(detail)
            seqs.setdefault(cid, []).append(
                (pl.get("attempt"), pl.get("reason"), dt.group(1) if dt else None, parsed))
        for cid, seq in seqs.items():
            dts = {s[2] for s in seq if s[2]}
            if len(seq) < 2 or len(dts) < 2:
                continue
            found_any = True
            print(f"  {run.name[4:]:<26} {cid}")
            for attempt, reason, dt, parsed in seq:
                if parsed:
                    cand, floor = parsed
                    where = f"cand={cand:.6f} floor={floor:.6f} {cand - floor:+.4f}"
                else:
                    where = "(no frac in detail -- compile/resource failure)"
                print(f"    attempt {attempt}  dtype={str(dt):<5} {str(reason):<24} {where}")
    if not found_any:
        print("  none")
    else:
        print("\n  A correctness rejection with no reproducible logic bug leaves the repair agent")
        print("  one lever it can always pull: precision. On this hardware that is never free --")
        print("  fp16->tf32 doubles staged bytes (and the space often has no shared-memory bound,")
        print("  so it OOMs), while tf32->fp16 INCREASES the deviation it was told to fix.")
        print("  See docs/finding-floor-rejection-sends-repair-after-the-dtype.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
