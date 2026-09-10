"""Given that all 537 are the candidate's fault, is the MULTIPLIER nonetheless mis-chosen?

The previous audit settled attribution: the fp64 relative arm ran on all 537 and refused all 537,
with a minimum ratio of 3.04x the reference's own error. So no candidate was refused by a threshold
it was inside of. That closes "the gate rejects correct kernels" for this corpus.

But it leaves one narrower question, and it IS a gate question: `_computes_low_precision` picks
which multiplier applies (2.0 for fp32-class, 3.0 for fp16/bf16), by reading the materialized PARAMS
literal. If it mis-reads a low-precision candidate as fp32-class, the candidate is judged at 2.0
where 3.0 was intended -- and anything landing in [2.0*ref, 3.0*ref] would be a genuine gate defect.

So measure three things:

  1 Which multiplier was actually applied, against what the trial's own PARAMS say the compute
    precision was. A disagreement is a defect in the detector, not in the candidate.
  2 How many trials fall in the band between the two multipliers -- i.e. would the verdict flip if
    the other multiplier had been chosen? That is the whole size of the exposure.
  3 The margin distribution near the threshold: how many sit within, say, 1.5x of their own
    threshold, which is where a mis-chosen multiplier could matter at all.

If (2) is zero the multiplier question is closed too, whatever (1) says -- a mis-detected precision
that changes no verdict is a latent defect, worth recording, not a cause of lost candidates.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_LOW = ("fp16", "bf16", "float16", "bfloat16", "half")


def _num(text: str, key: str) -> float | None:
    m = re.search(rf"'{key}':\s*'?([-+0-9.eE]+)'?", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def params_precision(values: dict) -> str:
    """What the trial's OWN materialized params say about compute precision.

    Reads the recorded param values rather than the source text, which is the same thing
    `_computes_low_precision` claims to do -- so a disagreement between this and the applied
    multiplier is a bug in that function.
    """
    found = set()
    for k, v in values.items():
        s = str(v).lower()
        ku = k.upper()
        if any(h in ku for h in ("PRECISION", "DTYPE", "PREC", "ACCUM", "ACC_TYPE")):
            found.add(s)
    if not found:
        return "unknown"
    if any(f in _LOW for f in found):
        return "low"
    if found & {"tf32", "ieee", "fp32", "float32", "tf32x3"}:
        return "fp32class"
    return "unknown"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    runs = sorted(Path(sys.argv[1]).glob("run-l3-*"))

    applied = defaultdict(int)
    cross = defaultdict(int)          # (params precision, applied multiplier) -> n
    band = 0                          # would flip if the other multiplier were used
    near = defaultdict(int)
    rows = []

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
            if tr.get("failure_kind") != "correctness_mismatch":
                continue
            d = tr.get("failure_detail") or ""
            if "fp64-relative arm ALSO failed" not in d:
                continue
            mult = _num(d, "multiplier")
            ratio = _num(d, "ratio_to_reference")
            if mult is None or ratio is None:
                continue
            vals = (tr.get("params") or {}).get("values") or {}
            prec = params_precision(vals)
            applied[mult] += 1
            cross[(prec, mult)] += 1
            # Would the OTHER multiplier have accepted it?
            other = 3.0 if mult == 2.0 else 2.0
            if ratio <= other:
                band += 1
                rows.append((run.name[:26], tr.get("candidate_id", "")[:14], prec, mult,
                             other, ratio))
            for b in (1.5, 2.0, 3.0, 5.0):
                if ratio <= b:
                    near[b] += 1

    total = sum(applied.values())
    print("failing trials whose fp64 arm reported its multiplier: %d\n" % total)

    print("=== 1. which multiplier was applied, vs what the trial's own params say ===")
    print("%-14s %-14s %-9s %s" % ("params say", "multiplier", "count", "agrees?"))
    print("-" * 56)
    for (prec, mult), n in sorted(cross.items()):
        expected = 3.0 if prec == "low" else (2.0 if prec == "fp32class" else None)
        agree = "n/a" if expected is None else ("yes" if expected == mult else "**NO**")
        print("%-14s %-14.1f %-9d %s" % (prec, mult, n, agree))
    print()

    print("=== 2. would the OTHER multiplier have changed the verdict? ===")
    print("trials that would be ACCEPTED under the other multiplier: %d (%.1f%%)"
          % (band, 100.0 * band / max(1, total)))
    if rows:
        print()
        for r in rows[:15]:
            print("   %-26s %-14s params=%-10s applied %.1f, other %.1f, ratio %.3f" % r)
    print()

    print("=== 3. margin distribution: ratio_to_reference ===")
    print("%-10s %-9s %s" % ("<= x", "count", "share"))
    print("-" * 34)
    for b in (1.5, 2.0, 3.0, 5.0):
        print("%-10.1f %-9d %.1f%%" % (b, near[b], 100.0 * near[b] / max(1, total)))
    print()
    print("The multipliers in force are 2.0 (fp32-class) and 3.0 (low precision). A count of 0 at")
    print("<=3.0 means no failing candidate was within EITHER multiplier, so no threshold choice")
    print("between them could have admitted it -- the multiplier question is closed for this")
    print("corpus regardless of what section 1 says about the detector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
