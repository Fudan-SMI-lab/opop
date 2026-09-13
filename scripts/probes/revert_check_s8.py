"""Revert-check for the S8 tests: does each guard fail when its behaviour is removed?

Same discipline as `revert_check_2e.py`, and the reason it exists for THIS change in particular:
optuna deprecated `categorical_distance_func` in 4.9.0 and removes it in 5.0.0, so the failure mode
worth fearing is an argument accepted and silently ignored -- a treatment arm byte-identical to the
control, with nothing in the log to say so. A test suite that passes on the reverted code would not
catch that.

Run from the v3 worktree root:  python scripts/probes/revert_check_s8.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

OD = "src/kernel_optimizer/tuning/ordered_domains.py"
TPE = "src/kernel_optimizer/tuning/tpe.py"
CFG = "src/kernel_optimizer/config.py"
ORCH = "src/kernel_optimizer/control/orchestrator.py"

# (label, file, old, new, what must break)
VARIANTS = [
    ("the distance never reaches the sampler", TPE,
     "            categorical_distance_func=distance,\n",
     "",
     "the switch would flip, the snapshot would be journalled, and sampling would not change"),

    ("the switch is ignored and the distance always applied", TPE,
     "        distance = (ordered_domains.distance_funcs(list(space.domains)) or None\n"
     "                    if ordered_categoricals else None)",
     "        distance = ordered_domains.distance_funcs(list(space.domains)) or None",
     "a run with S8 off would not be comparable with the finished ones"),

    ("an all-unordered space passes an empty dict instead of None", TPE,
     "        distance = (ordered_domains.distance_funcs(list(space.domains)) or None\n"
     "                    if ordered_categoricals else None)",
     "        distance = (ordered_domains.distance_funcs(list(space.domains))\n"
     "                    if ordered_categoricals else None)",
     "the deprecated optuna argument would be exercised for a call that asked for nothing"),

    ("the predicate trusts the agent-declared kind", OD,
     "    nums: list[float] = []\n"
     "    for c in values:\n"
     "        n = _as_num(c)\n"
     "        if n is None:\n"
     "            return None\n"
     "        nums.append(n)",
     "    if domain.kind not in (\"int\", \"float\"):\n"
     "        return None\n"
     "    nums = [float(c) for c in values]",
     "a kind=str knob holding numeric values would lose its order, and a mislabelled "
     "kind=int knob holding names would gain a meaningless one"),

    ("booleans become an ordered axis", OD,
     "    if isinstance(value, bool):\n        return None\n",
     "",
     "float(True) is 1.0, so an on/off switch would be handed a gradient"),

    ("two-value knobs get a distance", OD,
     "MIN_ORDERED_CHOICES = 3",
     "MIN_ORDERED_CHOICES = 2",
     "one gap cannot distinguish a smooth axis from two isolated labels"),

    ("a duplicated choice is accepted", OD,
     "    if len(set(nums)) != len(nums):\n        return None\n",
     "",
     "two identical rungs make two DIFFERENT labels distance-zero, i.e. interchangeable"),

    ("the distance uses raw values instead of rungs", OD,
     "        ra, rb = rung.get(_as_num(a)), rung.get(_as_num(b))\n"
     "        if ra is None or rb is None:\n"
     "            return float(len(rung))\n"
     "        return float(abs(ra - rb))",
     "        na, nb = _as_num(a), _as_num(b)\n"
     "        if na is None or nb is None:\n"
     "            return float(len(rung))\n"
     "        return float(abs(na - nb))",
     "on a geometric ladder 64->128 would read as eight steps and 16->32 as one"),

    ("the rungs follow the declared order instead of sorting", OD,
     "    rung = {value: i for i, value in enumerate(sorted(nums))}",
     "    rung = {value: i for i, value in enumerate(nums)}",
     "a shuffled choice list would make 16 and 128 adjacent"),

    ("an unknown value reads as identical", OD,
     "            return float(len(rung))",
     "            return 0.0",
     "an enqueued anchor's unseen value would be declared identical to everything"),

    ("unordered knobs get a constant distance instead of being omitted", OD,
     "    out: dict[str, Any] = {}\n"
     "    for domain in domains:\n"
     "        fn = rung_distance_for(domain)\n"
     "        if fn is not None:\n"
     "            out[domain.name] = fn\n"
     "    return out",
     "    out: dict[str, Any] = {}\n"
     "    for domain in domains:\n"
     "        fn = rung_distance_for(domain)\n"
     "        out[domain.name] = fn if fn is not None else (\n"
     "            lambda a, b: 0.0 if a == b else 1.0)\n"
     "    return out",
     "a precision switch would move into the distance-kernel path, which the probe measured "
     "to differ from optuna's default even when every distance is equal"),

    ("S8 defaults ON", CFG,
     "    enabled: bool = False\n\n\nclass V3Config(StrictConfig):",
     "    enabled: bool = True\n\n\nclass V3Config(StrictConfig):",
     "a run would silently stop being comparable with the finished ones"),

    ("the snapshot is journalled even when the switch is off", ORCH,
     "        ordered = (ordered_domains.snapshot(list(space.domains))\n"
     "                   if self.cfg.v3.ordered_categoricals.enabled else None)",
     "        ordered = ordered_domains.snapshot(list(space.domains))",
     "a run without S8 would gain a new key, so 'nothing qualified' and 'not asked' would merge"),
]


def run_tests():
    """The S8 suite plus the tuner's own, in the project environment.

    `uv run --offline --extra test` and not `sys.executable`: the interpreter running this script has
    no optuna on the Windows host, so every variant would report CAUGHT for the same unrelated import
    error -- a revert-check that always says CAUGHT proves nothing.
    """
    p = subprocess.run(["uv", "run", "--offline", "--extra", "test", "--quiet",
                        "python", "-m", "pytest",
                        "tests/test_s8_ordered_categoricals.py", "tests/test_tpe.py",
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
