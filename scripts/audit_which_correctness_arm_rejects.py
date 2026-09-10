"""Which correctness ARM does the rejecting? The frac gate is below this task's own noise floor.

Recorded: all three L3 tasks have a reference ieee-vs-tf32 frac below the 0.99 gate
(0.9554/0.9767/0.9778), so the frac arm can reject a candidate that is MORE self-consistent than
the reference. The fp64 relative arm was added for exactly that and is measured un-cheatable
(rescues had ratio < 1.0; adversarial defects rejected at 28-290x).

The question that matters: is any rejection carried by the frac arm ALONE? Those are the potential
false rejections. A rejection where BOTH arms fail is sound, because the fp64 arm compares the
candidate's rmse-vs-fp64 against the REFERENCE's own rmse-vs-fp64 and is therefore immune to the
floor.

A `SPACE_REJECTED` is NOT necessarily a correctness rejection: a space can die on a compile error,
a materialize error or an out-of-range witness, none of which reach either arm. My first version
counted those as "frac arm alone" -- one CompilationError in the corpus was reported as a false
rejection when the kernel never ran at all. So a row is only counted once it carries an actual
`frac_within_tol` reading, and anything else is reported separately rather than silently binned.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

d = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(d).name
both = frac_only = 0
rows = []
not_a_correctness_rejection = []
for l in open(pathlib.Path(d) / "events.jsonl", encoding="utf-8"):
    e = json.loads(l)
    t = e["type"]
    p = e.get("payload") or {}
    if t == "TRIAL_DONE":
        tr = p.get("trial") or {}
        if tr.get("failure_kind") != "correctness_mismatch":
            continue
        det = tr.get("failure_detail") or ""
        vals = ((tr.get("params") or {}).get("values")) or {}
    elif t == "SPACE_REJECTED":
        det = p.get("detail") or ""
        vals = {}
    else:
        continue
    if not det:
        continue
    fm = re.search(r"'frac_within_tol': ([0-9.]+)", det)
    if not fm:
        # A compile error, a materialize error, an agent failure -- it never reached either
        # correctness arm, so it is evidence about neither.
        kind = "compile_error" if "CompilationError" in det else (
            "runtime_error" if "runtime_error" in det else "other")
        not_a_correctness_rejection.append((t, kind))
        continue
    frac = float(fm.group(1))
    fp64_failed = "fp64-relative arm ALSO failed" in det
    m = re.search(r"'ratio_to_reference': '?([0-9.]+)", det)
    ratio = float(m.group(1)) if m else None
    if fp64_failed:
        both += 1
    else:
        frac_only += 1
    rows.append((t, vals.get("COMPUTE_DTYPE"), frac, ratio, fp64_failed))

print("%s  rejections that actually reached a correctness arm: %d" % (tag, len(rows)))
print("   BOTH arms failed (sound -- fp64 arm is immune to the floor): %d" % both)
print("   FRAC ARM ALONE (candidate may be more self-consistent than the ref): %d" % frac_only)
for r in rows:
    print("      %-14s %-5s frac=%-9s fp64_ratio=%-8s %s" % (
        r[0], r[1] or "-", r[2], r[3], "both" if r[4] else "FRAC-ONLY"))
if not_a_correctness_rejection:
    print("   (excluded, never reached a correctness arm: %d -- %s)" % (
        len(not_a_correctness_rejection),
        ", ".join("%s/%s" % x for x in not_a_correctness_rejection[:6])))
