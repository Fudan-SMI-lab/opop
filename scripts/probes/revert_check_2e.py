"""Revert-check for the 2e tests: does each guard actually fail when its behaviour is removed?

A test suite that passes on the reverted code proves nothing. For each variant this applies one
minimal edit that removes one guarded behaviour, runs the suite, and requires at least one FAILURE.
Restores the file afterwards, always.

Run from the v3 worktree root:  python scripts/probes/revert_check_2e.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

WA = "src/kernel_optimizer/evaluation/wall_attribution.py"
ORCH = "src/kernel_optimizer/control/orchestrator.py"
CFG = "src/kernel_optimizer/config.py"
REPORT = "src/kernel_optimizer/reporting/wall_report.py"

# (label, file, old, new, what must break)
VARIANTS = [
    ("categorical domains become ordered", WA,
     "    if isinstance(x, bool):\n        return None\n",
     "",
     "a bool in a mixed domain would be read as 0.0 and reported as a low wall"),

    ("the slope filter is removed", WA,
     "    worth = [w for w in walls if w.monotone and w.tail_gain_pct > 0.0]",
     "    worth = list(walls)",
     "walls whose latency worsens toward them would be probed"),

    ("an unanswered probe reads as innocent", WA,
     '    if fits is None:\n        return "undecidable"',
     '    if fits is None:\n        return "not_attributed"',
     "a worker timeout would count as evidence the knob is not the cause"),

    ("the prompt drops the at-the-optimum condition", WA,
     '    lines = ["共享内存墙(实测,非估计 —— 以下结论仅在该候选的最优参数点处成立):"]',
     '    lines = ["共享内存墙(实测):"]',
     "the conditional reading would be lost"),

    ("unattributed walls reach the prompt", WA,
     '    rows = [w for w in walls if w.verdict == "attributed"]',
     "    rows = list(walls)",
     "a guess would be presented as a measurement"),

    ("the two-value floor is removed", WA,
     "        if len(ran) < _MIN_RAN_VALUES:",
     "        if len(ran) < 2:",
     "a two-point knob would always read as monotone"),

    ("the farthest refused value is reported", WA,
     "            target = min(above)",
     "            target = max(above)",
     "the reported step would not be the smallest one that hits the wall"),

    ("2e defaults ON", CFG,
     "    enabled: bool = False\n\n    # Diagnostic depth",
     "    enabled: bool = True\n\n    # Diagnostic depth",
     "a run would spend GPU time without being asked"),

    ("the prompt path defaults ON", CFG,
     "    in_prompt: bool = False",
     "    in_prompt: bool = True",
     "what the agent sees would change without a control arm"),

    ("theta* selects on the mean", ORCH,
     "            ms = t.latency_ms.robust_ms",
     "            ms = t.latency_ms.mean",
     "the ablation origin could differ from the framework's own winner"),

    ("walls are probed one per process", ORCH,
     "        self.deps.evaluator.prescreen_batch(\n"
     "            self.task, [p for _, _, p in variants],\n"
     '            tag=f"{crun.candidate.candidate_id}-wall", backend=backend)',
     "        for _, _, _p in variants:\n"
     "            self.deps.evaluator.compile_screen(\n"
     '                self.task, _p, tag="wall", backend=backend,\n'
     "                max_shared_bytes=limit)",
     "the diagnostic would cost more than the waste it describes"),

    ("the diagnostic can end a candidate", ORCH,
     "        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never end a candidate\n"
     "            self.store.append(\"RESOURCE_WALL_ATTRIBUTION_FAILED\", {",
     "        except ValueError as exc:\n"
     "            self.store.append(\"RESOURCE_WALL_ATTRIBUTION_FAILED\", {",
     "a bookkeeping defect would surface as a candidate defect"),

    ("the report drops the object-shaped reader", REPORT,
     '        t = ev.get("type") if isinstance(ev, dict) else getattr(ev, "type", None)\n'
     '        if t != "RESOURCE_WALL_ATTRIBUTED":\n'
     "            continue\n"
     '        p = (ev.get("payload") if isinstance(ev, dict) '
     'else getattr(ev, "payload", None)) or {}',
     '        if not isinstance(ev, dict) or ev.get("type") != "RESOURCE_WALL_ATTRIBUTED":\n'
     "            continue\n"
     '        p = ev.get("payload") or {}',
     "report.py passes objects, so the section would silently be empty"),
]


def run_tests():
    env = dict(os.environ, PYTHONPATH="src")
    p = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/test_2e_wall_attribution.py", "-q", "--no-header", "-x"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return p.returncode, p.stdout.decode("utf-8", "replace")


code, out = run_tests()
print(f"BASELINE: {'PASS' if code == 0 else 'FAIL'}  {out.strip().splitlines()[-1]}")
if code != 0:
    sys.exit("baseline must pass before reverting anything")

results = []
for label, path, old, new, why in VARIANTS:
    text = io.open(path, encoding="utf-8").read()
    n = text.count(old)
    if n != 1:
        results.append((label, "PATCH-NOT-UNIQUE", f"pattern found {n} times", why))
        print(f"\n[{label}]\n  !! pattern found {n} times -- variant not applied, "
              f"so its 'pass' would be meaningless")
        continue
    backup = tempfile.mktemp(suffix=".bak")
    shutil.copy2(path, backup)
    try:
        io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(old, new, 1))
        code, out = run_tests()
        last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        verdict = "CAUGHT" if code != 0 else "NOT CAUGHT"
        results.append((label, verdict, last, why))
        print(f"\n[{label}]\n  {verdict}: {last}\n  guards: {why}")
    finally:
        shutil.copy2(backup, path)
        os.unlink(backup)

print("\n" + "=" * 78)
bad = [r for r in results if r[1] != "CAUGHT"]
for label, verdict, last, why in results:
    print(f"  {verdict:18s} {label}")
print(f"\n{len(results) - len(bad)}/{len(results)} variants caught")
if bad:
    print("UNGUARDED BEHAVIOUR -- each of these can be removed with the suite still green:")
    for label, verdict, last, why in bad:
        print(f"  - {label}: {why}")

code, out = run_tests()
print(f"\nRESTORED: {'PASS' if code == 0 else 'FAIL'}  {out.strip().splitlines()[-1]}")
sys.exit(1 if bad or code != 0 else 0)
