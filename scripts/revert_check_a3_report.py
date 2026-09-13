"""Revert check for the A3 report section: break each claim, confirm a test catches it.

A test that passes whether or not the code works is worse than no test, and this project has been
bitten by exactly that four times (a test that copied the loop, a variant that changed no behaviour,
a fixture invented to match the reader, an unreachable branch). So each mutation below removes ONE
behaviour the new tests claim to pin, and the check is that at least one test FAILS. A mutation that
leaves the suite green names a hole in the tests, not a redundancy in the code.

Run: python scripts/revert_check_a3_report.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "kernel_optimizer" / "reporting" / "wall_report.py"
TESTS = "tests/test_2e_wall_attribution.py"

# (name, pattern, replacement) -- each undoes one decision the section makes.
MUTATIONS = [
    (
        "float/int comparison: require exact equality",
        r"    try:\n        return float\(a\) == float\(b\)\n    except \(TypeError, ValueError\):\n        return a == b",
        "    return a == b",
    ),
    (
        "report the footprint from EVERY completed trial, not only those at the refused value",
        r"                if not _same_value\(vals\[knob\], refused\):\n                    continue",
        "                pass",
    ),
    (
        "collapse 'dimension gone' into 'not tried'",
        r'            elif not saw_knob:\n                lines\.append\(f"\| `\{cid\}` \| \{knob\} \| \*\*\{_fmt_val\(refused\)\}\*\* \| `\{kid\}` \| "\n                             f"维度已消失\(改写后无此 knob\) \| —— \| \{limit\} \|"\)',
        '            elif not saw_knob:\n                lines.append(f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | `{kid}` | "\n                             f"未曾尝试(TPE 非均匀采样,缺席≠被拒) | —— | {limit} |")',
    ),
    (
        "drop the 'no descendant yet' row",
        r'        if not kids:\n            lines\.append\(\n                f"\| `\{cid\}` \| \{knob\} \| \*\*\{_fmt_val\(refused\)\}\*\* \| —— \| "\n                f"\*\*该墙尚无后代\*\*\(改写未产生/未注册\) \| —— \| \{limit\} \|"\)\n            any_row = True\n            continue',
        "        if not kids:\n            continue",
    ),
    (
        "render A3 for unattributed walls too",
        r'            if w\.get\("verdict"\) == "attributed":\n                attributed\.append\(\(str\(p\.get\("candidate_id"\) or "\?"\), w\)\)',
        '            attributed.append((str(p.get("candidate_id") or "?"), w))',
    ),
    (
        "drop the counterfactual caveat",
        r'    lines\.append\(\n        "\*\*这一节不能单独证明 2e 有效\*\*.*?a3_counterfactual_arm_without_2e\.py`\)。"\)',
        "    pass",
    ),
    (
        "only walk direct children, not the whole lineage",
        r"        out: list\[str\] = \[\]\n        stack = list\(children\.get\(cid, \[\]\)\)\n        while stack:\n            x = stack\.pop\(0\)\n            out\.append\(x\)\n            stack\.extend\(children\.get\(x, \[\]\)\)\n        return out",
        "        return list(children.get(cid, []))",
    ),
    (
        "never say the footprint fell below the limit",
        r'                under = \(isinstance\(limit, \(int, float\)\) and real and max\(real\) <= limit\)',
        "                under = False",
    ),
    (
        "stop appending A3 to the section at all",
        r"    lines\.extend\(freed_lines\(events\)\)",
        "    pass",
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
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
            cwd=ROOT, capture_output=True,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
        out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
        if proc.returncode == 0:
            results.append((name, "*** SUITE STILL GREEN -- the tests do not cover this"))
        else:
            first = ""
            for ln in out.splitlines():
                if ln.startswith("FAILED") or "::" in ln and "Error" in ln:
                    first = ln.strip()
                    break
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
