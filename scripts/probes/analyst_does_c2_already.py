"""Can the ANALYST already do what C2 claims -- find a range truncated by a resource limit and argue
from the slope -- without 2e?

WHY THIS THREATENS THE CONTRIBUTION. C2 is "find the steepest-slope dimension whose range is truncated
by a resource limit, and steer the structural rewrite to free it". The control arm has
`slope_guide.enabled: False` and zero attributed walls, yet its analyst wrote:

    "Unblocks the shared-capped directions: NUM_STAGES 6-8 and/or BLOCK_N=256 at 4 stages fit under the
     101376 B cap (BN=256 bf16 tiles s4 = 98304 B exactly)"
    "Picks up the boundary trends now blocked: stages 1->4 gave -25% at the winning tile; BLOCK_N 32->128
     was monotone -24%"

That is the C2 reasoning, arrived at without 2e. And it is not an accident: the analyst's own prompt
(modules.py:1059-1088) hands it `at_boundary` + direction, effect sizes, per-value failure rates,
resource usage at the best point, and `docs/device.md`'s hard limits, then asks it explicitly whether a
direction is "prevented by a hardware/resource limit". So the baseline agent is INSTRUCTED to do this.

BE-CBO was an external precedent for the boundary observation; this is worse, because it is OUR OWN
baseline. If the analyst reaches the same conclusions unaided, 2e's marginal contribution is not a new
capability -- it would have to be earlier, more reliable, or quantitatively better, and each of those is
a different (and measurable) claim.

WHAT THIS MEASURES. Per arm, over every analyst hypothesis: does its text (a) name a resource limit as
the blocker, (b) cite a slope/trend, (c) do both -- the C2 shape? Counting BOTH arms matters: if the
treatment arm's analyst does it at the same rate, 2e added nothing to the analyst either.

The classifier is deliberately crude and its terms are printed, because the interesting number is a
RATE, not a verdict on any one hypothesis -- and a hand-tuned matcher on 20 hypotheses would be fitting
noise. Every match is printed so the reading can be checked rather than trusted.

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/analyst_does_c2_already.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")

# A resource LIMIT named as the thing blocking a direction. Not just "shared memory" -- the claim is
# about a cap being hit, so the terms are about capacity and blocking.
_LIMIT = re.compile(
    r"\b(cap|capped|limit|limited|blocked|blocking|unblock|does not fit|doesn't fit|fits under|"
    r"exceed|over the|101376|166912|49152|out of resource|shared[- ]capped|register[- ]limited|"
    r"occupancy[- ]limited)\b", re.I)

# A SLOPE or trend argument: a measured direction of improvement across a knob's values.
_SLOPE = re.compile(
    r"\b(monotone|monotonic|trend|boundary|slope|gave -?\d+(\.\d+)?%|-\d+(\.\d+)?% at|"
    r"\d+\s*->\s*\d+|improv\w+ toward|best single value|at_boundary)\b", re.I)


def main() -> int:
    print("LIMIT terms: %s" % _LIMIT.pattern[:120])
    print("SLOPE terms: %s" % _SLOPE.pattern[:120])
    print()
    totals: dict[str, list[int]] = {}
    for run in sorted(BASE.glob("s7-*/run-l3-43-*")):
        arm = run.parent.name
        rows = [0, 0, 0, 0]  # hypotheses, limit-only, slope-only, BOTH
        shown = 0
        for line in (run / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") != "BOTTLENECK_REPORTED":
                continue
            p = e.get("payload") or {}
            rep = p.get("report") or p
            for h in (rep.get("hypotheses") or []):
                if not isinstance(h, dict):
                    continue
                text = " ".join(str(h.get(k) or "") for k in ("change", "expected_effect", "risk"))
                if not text.strip():
                    continue
                rows[0] += 1
                lim, slo = bool(_LIMIT.search(text)), bool(_SLOPE.search(text))
                if lim and slo:
                    rows[3] += 1
                    if shown < 3:
                        shown += 1
                        print("  %s %s/%s -- BOTH:" % (arm, p.get("candidate_id"), h.get("id")))
                        for m in list(_LIMIT.finditer(text))[:2]:
                            print("      limit: ...%s..." % text[max(0, m.start() - 60):m.end() + 60])
                        for m in list(_SLOPE.finditer(text))[:2]:
                            print("      slope: ...%s..." % text[max(0, m.start() - 60):m.end() + 60])
                elif lim:
                    rows[1] += 1
                elif slo:
                    rows[2] += 1
        totals[arm] = rows

    print()
    print("%-14s %12s %12s %12s %14s" % ("arm", "hypotheses", "limit only", "slope only", "BOTH (C2)"))
    for arm, r in sorted(totals.items()):
        print("%-14s %12d %12d %12d %8d (%.0f%%)" % (
            arm, r[0], r[1], r[2], r[3], 100 * r[3] / r[0] if r[0] else 0.0))

    print()
    ctrl = totals.get("s7-control", [0, 0, 0, 0])
    if ctrl[3] > 0:
        print("THE CONTROL ARM'S ANALYST PRODUCES THE C2 SHAPE %d TIME(S) with slope_guide DISABLED and"
              % ctrl[3])
        print("zero attributed walls. So 'name the resource-truncated direction and argue from the")
        print("slope' is NOT a capability 2e adds -- the baseline analyst is explicitly asked for it")
        print("(modules.py:1079) and is given at_boundary, effect sizes, resource usage at the best")
        print("point, and the device's hard limits. Any C2 claim has to be about being EARLIER, more")
        print("RELIABLE, or QUANTITATIVELY better than this -- each a different, measurable claim, and")
        print("none of them 'the framework could not do this before'.")
    else:
        print("The control arm's analyst did NOT produce the C2 shape in this corpus. That is the")
        print("result C2 needs, but check the term lists above before relying on it -- a crude matcher")
        print("missing the phrasing looks identical to the analyst not doing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
