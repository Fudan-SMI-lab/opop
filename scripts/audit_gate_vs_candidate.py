"""Of the 537 correctness_mismatch trials, which are the GATE's fault and which the candidate's?

The distinction is measurable and does not need to be argued, because the failure detail the worker
writes carries the numbers both arms actually used:

    relaxed mismatch on trial N; gate needs frac_within_tol>0.99 AND cosine>=0.99985
      vs ieee ref: {...}
      vs tf32 ref: {...}
      reference's OWN ieee-vs-tf32 spread (task noise floor, NOT a bug): {...}
      fp64-relative arm ALSO failed (...): {reference_rmse_vs_fp64, candidate_rmse_vs_fp64, ...}

So for each failing trial ask, in this order:

  A  Is the candidate's error at or BELOW the reference's own ieee-vs-tf32 spread?
     If so the candidate is inside the noise the reference itself produces, and the ABSOLUTE gate
     refused something no measurement on this task can distinguish from correct. GATE.

  B  Did the fp64 relative arm run at all?
     If it never ran -- switch off, or golden model unavailable -- then the only judgement made was
     the absolute one, which all three L3 tasks' floors sit below. GATE (by omission).

  C  The fp64 arm ran and refused it. Then the candidate is measurably further from fp64 TRUTH than
     the multiplier allows. CANDIDATE -- and the known root causes are on record: an uncompensated
     `dot`, or PREC gating the whole algorithm so a non-fp16 choice falls into an unstabilised
     branch.

The point of separating them: B and C have opposite fixes. Widening a threshold to admit C is
reward hacking; leaving B unfixed throws away correct kernels.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_FLOAT = r"[-+0-9.eE]+"


def _num(text: str, key: str) -> float | None:
    m = re.search(rf"'{key}':\s*'?({_FLOAT})'?", text)
    if not m:
        m = re.search(rf'"{key}":\s*"?({_FLOAT})"?', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _section(detail: str, label: str) -> str:
    """The metrics dict printed after `label`, up to the end of that line."""
    idx = detail.find(label)
    if idx < 0:
        return ""
    return detail[idx: detail.find("\n", idx) if detail.find("\n", idx) > 0 else len(detail)]


def classify(detail: str) -> tuple[str, dict]:
    """-> (verdict, numbers). Verdicts: gate_noise_floor, gate_no_fp64_arm, candidate, shape,
    unparsed."""
    if not detail:
        return "unparsed", {}
    if "shape mismatch" in detail:
        return "shape", {}

    ieee = _section(detail, "vs ieee ref:")
    tf32 = _section(detail, "vs tf32 ref:")
    floor = _section(detail, "reference's OWN")
    fp64 = _section(detail, "fp64-relative arm ALSO failed")

    out: dict = {}
    # frac_within_tol is the criterion the absolute gate applies; take the candidate's BEST arm,
    # because the gate accepts either witness.
    cand_frac = max([v for v in (_num(ieee, "frac_within_tol"), _num(tf32, "frac_within_tol"))
                     if v is not None] or [float("nan")])
    floor_frac = _num(floor, "frac_within_tol")
    out["candidate_frac"] = cand_frac
    out["floor_frac"] = floor_frac

    if fp64:
        ref_rmse = _num(fp64, "reference_rmse_vs_fp64")
        cand_rmse = _num(fp64, "candidate_rmse_vs_fp64")
        out["reference_rmse"] = ref_rmse
        out["candidate_rmse"] = cand_rmse
        if ref_rmse and cand_rmse and ref_rmse > 0:
            out["rmse_ratio"] = cand_rmse / ref_rmse
        # The fp64 arm RAN and refused: that is a judgement against truth, so it is the
        # candidate's, whatever the absolute gate thought.
        return "candidate", out

    if "fp64-relative arm unavailable" in detail:
        return "gate_no_fp64_arm", out

    # No fp64 section at all -> the arm never ran on this trial (switch off, or the absolute gate
    # was the only thing consulted).
    if floor_frac is not None and cand_frac == cand_frac:  # not NaN
        # A candidate at or above the reference's own agreement is inside task noise.
        if cand_frac >= floor_frac:
            return "gate_noise_floor", out
        return "gate_no_fp64_arm", out
    return "unparsed", out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    runs = sorted(Path(sys.argv[1]).glob("run-l3-*"))
    if not runs:
        print("no run-l3-* under %s" % sys.argv[1])
        return 1

    verdicts: dict[str, int] = defaultdict(int)
    per_run: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ratios: list[float] = []
    noise_cases: list[tuple[str, float, float]] = []
    rescued_total = 0
    rescue_seen = 0

    for run in runs:
        ev = run / "events.jsonl"
        if not ev.exists():
            continue
        for line in ev.open(encoding="utf-8"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            # The fp64 arm's own counter, journalled per trial: how many of that trial's
            # correctness trials were accepted ONLY by the relative arm.
            r = tr.get("fp64_rescued_trials")
            if isinstance(r, int):
                rescue_seen += 1
                rescued_total += r
            if tr.get("failure_kind") != "correctness_mismatch":
                continue
            v, nums = classify(tr.get("failure_detail") or "")
            verdicts[v] += 1
            per_run[run.name][v] += 1
            if v == "candidate" and "rmse_ratio" in nums:
                ratios.append(nums["rmse_ratio"])
            if v == "gate_noise_floor":
                noise_cases.append((tr.get("candidate_id", "")[:16],
                                    nums.get("candidate_frac") or 0.0,
                                    nums.get("floor_frac") or 0.0))

    total = sum(verdicts.values())
    print("correctness_mismatch trials classified: %d\n" % total)
    print("%-22s %-9s %s" % ("verdict", "count", "share"))
    print("-" * 46)
    for v in ("candidate", "gate_noise_floor", "gate_no_fp64_arm", "shape", "unparsed"):
        n = verdicts.get(v, 0)
        print("%-22s %-9d %.1f%%" % (v, n, 100.0 * n / max(1, total)))
    print()

    gate = verdicts.get("gate_noise_floor", 0) + verdicts.get("gate_no_fp64_arm", 0)
    print("GATE-attributable : %d (%.1f%%)" % (gate, 100.0 * gate / max(1, total)))
    print("CANDIDATE         : %d (%.1f%%)" % (verdicts.get("candidate", 0),
                                               100.0 * verdicts.get("candidate", 0) / max(1, total)))
    print()

    if ratios:
        ratios.sort()
        print("For the CANDIDATE ones, how far past the multiplier are they?")
        print("  candidate_rmse / reference_rmse : min %.2f  median %.2f  p90 %.2f  max %.2f"
              % (ratios[0], ratios[len(ratios) // 2], ratios[int(len(ratios) * 0.9)], ratios[-1]))
        for bound in (3.0, 10.0, 100.0, 1000.0):
            n = sum(1 for r in ratios if r > bound)
            print("  beyond %7.0fx the reference's own error: %d (%.1f%%)"
                  % (bound, n, 100.0 * n / len(ratios)))
        print()
        print("  Read this against the multipliers actually applied (2.0 fp32-class, 3.0 low")
        print("  precision). A cluster just above 3 would mean the multiplier is the binding")
        print("  constraint; a cluster orders of magnitude out means the kernels are simply wrong.")
        print()

    if noise_cases:
        print("Trials the ABSOLUTE gate refused while inside the reference's own spread (%d):"
              % len(noise_cases))
        for cid, cand, floor in noise_cases[:10]:
            print("   %-18s candidate frac %.6f  vs task noise floor %.6f" % (cid, cand, floor))
        print()

    print("fp64 relative arm, from its own journalled counter:")
    print("  trials carrying the counter : %d" % rescue_seen)
    print("  correctness trials rescued by the relative arm alone : %d" % rescued_total)
    print()
    print("A zero here with the switch ON would mean the arm is present and never load-bearing.")
    print("A zero because the counter is absent means something different -- the arm never ran.")
    print()
    print("Per run:")
    for name, d in sorted(per_run.items()):
        print("  %-34s %s" % (name[:34], dict(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
