"""Would the new tests actually CATCH a regression in the no-wall diagnosis?

A green suite is not evidence a guard works -- the recorded `a-clean-run-is-not-evidence-a-checker-works`
trap, and this project has already shipped tests that passed on an empty list. So each variant below
breaks the diagnosis in one specific way and the suite must fail. A variant that leaves the correct
answer reachable through another branch is not a variant, so every one of these removes or inverts the
behaviour rather than adding a parallel path.

Run:  PYTHONIOENCODING=utf-8 uv run --offline --extra test python scripts/probes/revert_check_s7_reader.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TARGET = Path("scripts/analyze_s7_pair.py")
TESTS = "tests/test_s7_pair_reader.py"

# (name, pattern, replacement) -- each must make the suite fail.
VARIANTS: list[tuple[str, str, str]] = [
    (
        "the no-wall cause is never computed (always None)",
        r'    if out\["n_refusals"\] == 0:',
        '    if False:',
    ),
    (
        "missing input and no truncation are conflated (first branch swallows both)",
        r'    if out\["n_refusals"\] == 0:',
        '    if True:',
    ),
    (
        "the worthless-wall cause is dropped, so the slope filter reads as 'no wall existed'",
        r'    elif out\["walls_found_total"\] == out\["walls_worthless_total"\]:',
        '    elif False:',
    ),
    (
        "a found, worthy wall is still given an excuse (positive control must fire)",
        r'        out\["no_wall_cause"\] = None',
        '        out["no_wall_cause"] = "walls found but ALL worthless"',
    ),
    (
        "walls_found is not read from the payload, so every run looks wall-free",
        r'        out\["walls_found_total"\] \+= int\(p\.get\("walls_found"\) or 0\)',
        '        out["walls_found_total"] += 0',
    ),
    (
        "walls_worthless is not read, so a dropped wall reads as a kept one",
        r'        out\["walls_worthless_total"\] \+= int\(p\.get\("walls_worthless"\) or 0\)',
        '        out["walls_worthless_total"] += 0',
    ),
    (
        "the soft gate's applicability is hard-wired off",
        r'        if p\.get\("applicable"\):',
        '        if False:',
    ),
    (
        "the soft reason string is discarded",
        r'        r = str\(p\.get\("reason"\) or "\(none given\)"\)',
        '        r = "(none given)"',
    ),
    (
        "the instrument check is disabled (the ORIGINAL defect, reinstated)",
        r'    out\["instrument_on"\] = out\["n_wall_events"\] > 0',
        '    out["instrument_on"] = True',
    ),
]


def run_suite() -> tuple[bool, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q"],
        capture_output=True)
    out = (p.stdout + p.stderr).decode("utf-8", errors="replace")
    return p.returncode == 0, out


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")

    ok, out = run_suite()
    if not ok:
        print("BASELINE FAILS -- fix that before trusting any variant below")
        print(out[-2000:])
        return 2
    print("baseline: PASS\n")

    caught = missed = 0
    try:
        for name, pat, repl in VARIANTS:
            new, n = re.subn(pat, repl, original, count=1)
            if n != 1:
                print(f"  !! PATTERN DID NOT MATCH ({n} hits): {name}")
                print("     a variant that changes nothing proves nothing -- fix the pattern")
                missed += 1
                continue
            TARGET.write_text(new, encoding="utf-8")
            passed, _ = run_suite()
            if passed:
                print(f"  NOT CAUGHT: {name}")
                missed += 1
            else:
                print(f"  caught:     {name}")
                caught += 1
    finally:
        TARGET.write_text(original, encoding="utf-8")

    ok, _ = run_suite()
    print(f"\nrestored, baseline {'PASS' if ok else 'FAIL -- RESTORE DID NOT WORK'}")
    print(f"{caught} caught / {caught + missed} variants")
    return 0 if missed == 0 and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
