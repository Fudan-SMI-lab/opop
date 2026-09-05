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
                                   cand, floor, cand > floor))
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

    print("\n  ABOVE -- rejected although more consistent than the reference is with itself:")
    for task, cid, reason, cand, floor, _ in sorted(above, key=lambda r: -(r[3] - r[4])):
        print(f"    {task:<8} {str(cid):<16} {str(reason):<26} "
              f"cand={cand:.6f} floor={floor:.6f}  +{cand - floor:.4f}")

    print("\n  BELOW -- and note how thin the margins are; a floor-relative gate must pick a")
    print("  tolerance that decides these too:")
    for task, cid, reason, cand, floor, _ in sorted(below, key=lambda r: -(r[3] - r[4])):
        print(f"    {task:<8} {str(cid):<16} {str(reason):<26} "
              f"cand={cand:.6f} floor={floor:.6f}  {cand - floor:+.4f}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
