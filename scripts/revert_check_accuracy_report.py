"""Revert check for the 2b(2)/2d accuracy section: break each claim, confirm a test catches it.

Same discipline as `revert_check_a3_report.py`. Each mutation removes ONE decision the section makes;
a mutation that leaves the suite green names a hole in the TESTS, not redundancy in the code.

The trial-count mutation is the important one: that gate is what made `under-converged` structurally
dead in the first version, so a test must exist that fails when it comes back.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "kernel_optimizer" / "reporting" / "accuracy_report.py"
TESTS = "tests/test_2b2_2d_accuracy_report.py"

MUTATIONS = [
    (
        "bring back the trial-count gate that made under-converged unreachable",
        r'        under = bool\(cinfo and cinfo\.get\("late"\)\)',
        '        under = bool(cinfo and cinfo.get("n", 0) <= 20 and cinfo.get("late"))',
    ),
    (
        "let under-converged promote the miss to a hit",
        r'                    d\["under"\] \+= 1',
        '                    d["under"] += 1\n'
        '                    d["hit"] += 1\n'
        '                    d["miss"] -= 1',
    ),
    (
        "flag every dead lever as under-converged regardless of convergence",
        r'        under = bool\(cinfo and cinfo\.get\("late"\)\)',
        "        under = True",
    ),
    (
        "collapse the direction flip into the dead-lever count",
        r'            elif expected in \("up", "down"\) and actual in \("up", "down"\) and expected != actual:\n                d\["flip"\] \+= 1',
        '            elif False:\n                d["flip"] += 1',
    ),
    (
        "report an impossible miss shape as a plain zero finding",
        r'        if n_rounds and full == n_rounds:',
        "        if False:",
    ),
    (
        "count unmeasured declarations in the denominator",
        r'            if match in d:\n                d\[match\] \+= 1',
        '            if match in d:\n                d[match] += 1\n'
        '            if match == "unmeasured":\n                d["miss"] += 1',
    ),
    (
        "pool all dimensions into one row",
        r'            dim = str\(row\.get\("dimension"\) or "\?"\)',
        '            dim = "all"',
    ),
    (
        "drop the implementation-rate condition",
        r'    lines\.append\(\n        "\*\*每个准确率都只在「agent 愿意实现的那些假设」上成立。\*\* 实测 analyst 提出的假设只有 "',
        '    lines.append(\n        "" "',
    ),
    (
        "handle only dict events, not report.py's objects",
        r'def _ev\(e: Any, name: str\) -> Any:\n    return e\.get\(name\) if isinstance\(e, dict\) else getattr\(e, name, None\)',
        "def _ev(e: Any, name: str) -> Any:\n    return e.get(name) if isinstance(e, dict) else None",
    ),
]

original = TARGET.read_text(encoding="utf-8")
results = []
try:
    for name, pattern, repl in MUTATIONS:
        mutated, n = re.subn(pattern, repl, original, count=1, flags=re.DOTALL)
        if n != 1:
            results.append((name, "PATTERN DID NOT MATCH -- mutation not applied"))
            continue
        TARGET.write_text(mutated, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
            cwd=ROOT, capture_output=True, env=env)
        out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        if proc.returncode == 0:
            results.append((name, "*** SUITE STILL GREEN -- the tests do not cover this"))
        else:
            first = next((ln.strip() for ln in out.splitlines() if ln.startswith("FAILED")), "")
            results.append((name, f"caught{(' by ' + first) if first else ''}"))
finally:
    TARGET.write_text(original, encoding="utf-8")

print(f"restored {TARGET.relative_to(ROOT)}")
print()
caught = sum(1 for _, r in results if r.startswith("caught"))
for name, r in results:
    print(f"  {'OK  ' if r.startswith('caught') else 'GAP '} {name}\n        {r}")
print()
print(f"{caught}/{len(MUTATIONS)} mutations caught")
if caught != len(MUTATIONS):
    print("A gap means a test asserts something the code is not required to do. Fix the TEST.")
    sys.exit(1)
