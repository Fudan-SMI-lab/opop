"""Revert-check for the item-2 tests: does each guard fail when its behaviour is removed?

Same discipline as `revert_check_2e.py` and `revert_check_s8.py`. The behaviours most worth guarding
here are the ones whose removal produces a PLAUSIBLE output rather than an error: an applicability gate
that stops firing turns 103 structural non-events into 103 "walls found", and a caveat line that
disappears turns an unconfirmed observation into something a reader takes for a probed verdict.

Run from the v3 worktree root:  python scripts/probes/revert_check_item2.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SW = "src/kernel_optimizer/evaluation/soft_wall.py"
CFG = "src/kernel_optimizer/config.py"
ORCH = "src/kernel_optimizer/control/orchestrator.py"
REPORT = "src/kernel_optimizer/reporting/wall_report.py"

VARIANTS = [
    ("the applicability gate is removed", SW,
     "    if float(spills_at_best) <= 0.0:\n"
     "        # NOT a negative result. 32 of 56 measured winners land here.\n"
     "        scan.reason = \"the best trial does not spill (no soft wall exists at the optimum)\"\n"
     "        return scan\n",
     "",
     "103 of 153 candidates whose winner does not spill would be scanned anyway, turning a "
     "structural non-event into a wall count"),

    ("the gate reads any trial instead of the best", SW,
     "    best = min(done, key=lambda t: _robust_ms(t) or float(\"inf\"))",
     "    best = done[0]",
     "a candidate with one slow spilling trial would look walled at a point nobody ships"),

    ("an unmeasured n_spills counts as zero", SW,
     "    if not isinstance(spills_at_best, (int, float)) or isinstance(spills_at_best, bool):\n"
     "        scan.reason = \"the best trial has no n_spills measurement\"\n"
     "        return scan\n",
     "    if not isinstance(spills_at_best, (int, float)) or isinstance(spills_at_best, bool):\n"
     "        spills_at_best = 0\n",
     "an unmeasured field would be reported as a measurement of zero"),

    ("the monotone condition is dropped", SW,
     "        if not all(spills[i + 1] >= spills[i] for i in range(len(spills) - 1)):\n"
     "            continue\n",
     "",
     "the 27-of-56 peaked candidates would be forced into a monotone reading"),

    ("the slope filter is dropped", SW,
     "        if gain <= 0.0:\n            continue\n",
     "",
     "a knob whose latency WORSENS toward the spilling end would be reported as an opportunity"),

    ("the three-value floor is lowered", SW,
     "_MIN_VALUES = 3",
     "_MIN_VALUES = 2",
     "two points are monotone through any two points"),

    # The realistic version of this mistake is ORDINAL ENCODING -- numbering the choices in list order
    # -- not mapping them all to one constant. Mapping them to a constant collapses every categorical
    # knob into a single bucket, which the `_MIN_VALUES = 3` floor already rejects, so that variant
    # changes no behaviour and is not a variant at all (`a-variant-that-changes-no-behaviour-is-not-a-
    # variant`). Ordinal encoding produces three real buckets and a plausible slope over the order the
    # agent happened to write, which is the thing `_as_num` exists to refuse.
    ("categorical knobs get an ordinal encoding", SW,
     "        for knob, raw in _params(t).items():\n"
     "            v = _as_num(raw)\n"
     "            if v is not None:\n"
     "                buckets[str(knob)][v].append((float(sp), ms))",
     "        for knob, raw in _params(t).items():\n"
     "            v = _as_num(raw)\n"
     "            if v is None:\n"
     "                v = float(abs(hash(str(raw))) % 97)\n"
     "            buckets[str(knob)][v].append((float(sp), ms))",
     "a precision switch would get a slope over the order the agent happened to write"),

    ("a knob outside param_stats can produce a wall", SW,
     "    known = {ps.name for ps in stats.param_stats} or set(buckets)",
     "    known = set(buckets)",
     "a wall could name a knob the report never lists"),

    ("robust_ms is looked up by name instead of reproduced", SW,
     "    if isinstance(lat, dict):\n"
     "        for key in (\"median\", \"mean\"):",
     "    if isinstance(lat, dict):\n"
     "        for key in (\"robust_ms\",):",
     "robust_ms is a @property and never serialized, so every replayed trial would read as untimed"),

    ("the limiter is read under the property's name", SW,
     "        v = occ.get(\"limiter\")",
     "        v = occ.get(\"occupancy_limiter\")",
     "the emitter writes `limiter`; the long name is the model PROPERTY that reads it, so a "
     "measured field would report as unmeasured"),

    ("the prompt drops the no-probe caveat", SW,
     "    lines = [\"寄存器溢出墙(实测,来自逐 trial profile,**未经独立探针确认** —— \"\n"
     "             \"以下结论仅在该候选已测的参数点上成立):\"]",
     "    lines = [\"寄存器溢出墙(实测):\"]",
     "an unconfirmed observation would be taken for a probed verdict"),

    ("the payload drops probe_confirmed", SW,
     "            \"probe_confirmed\": False,\n",
     "",
     "a reader of the raw log never sees for_prompt, so the row must carry the distinction"),

    ("the soft wall defaults ON", CFG,
     "    enabled: bool = False\n\n    # Put the spill walls into the analyst/rewriter prompt.",
     "    enabled: bool = True\n\n    # Put the spill walls into the analyst/rewriter prompt.",
     "a run would gain an event and stop being byte-comparable without being asked"),

    ("the prompt path defaults ON", CFG,
     "    in_prompt: bool = False\n\n    # How many walls the prompt may carry.",
     "    in_prompt: bool = True\n\n    # How many walls the prompt may carry.",
     "what the agent sees would change without a control"),

    ("the scan is journalled only when it finds something", ORCH,
     "            scan = soft_wall.find_soft_walls(crun.stats, list(crun.trials))\n"
     "            self.store.append(\"RESOURCE_SOFT_WALL\", {\n"
     "                \"candidate_id\": crun.candidate.candidate_id,\n"
     "                **scan.payload(),\n"
     "            })",
     "            scan = soft_wall.find_soft_walls(crun.stats, list(crun.trials))\n"
     "            if scan.walls:\n"
     "                self.store.append(\"RESOURCE_SOFT_WALL\", {\n"
     "                    \"candidate_id\": crun.candidate.candidate_id,\n"
     "                    **scan.payload(),\n"
     "                })",
     "'nothing found' is a result, and its absence reads as the switch being off"),

    ("the diagnostic can end a candidate", ORCH,
     "        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never end a candidate\n"
     "            self.store.append(\"RESOURCE_SOFT_WALL_FAILED\", {",
     "        except ValueError as exc:\n"
     "            self.store.append(\"RESOURCE_SOFT_WALL_FAILED\", {",
     "a bookkeeping defect would surface as a candidate defect"),

    ("the report drops the soft section when 2e is off", REPORT,
     "    payloads = _rows(events)\n    if not payloads:\n        return soft_wall_lines(events)",
     "    payloads = _rows(events)\n    if not payloads:\n        return []",
     "a soft-wall-only run would render nothing at all"),

    ("the report drops the applicability denominators", REPORT,
     "    lines.append(\n"
     "        f\"- **不适用 {n - len(applicable)} 个,其中 {zero_spill} 个的最优 trial 根本不溢出** —— \"\n"
     "        \"这不是阴性结果:最优点不溢出时,任何 knob 都不可能在该点被溢出截断。\"\n"
     "        \"「不适用」与「适用但无墙」是两种状态,必须分开读。\")",
     "",
     "a wall count with no denominator cannot be read at all"),
]


def run_tests():
    p = subprocess.run(["uv", "run", "--offline", "--extra", "test", "--quiet",
                        "python", "-m", "pytest",
                        "tests/test_item2_soft_wall.py", "tests/test_2e_wall_attribution.py",
                        "-q", "--no-header", "-x"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
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
