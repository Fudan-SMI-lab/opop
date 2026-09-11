# -*- coding: utf-8 -*-
"""Revert-check the zero-is-a-measurement fix. Both variants must be CAUGHT."""
from __future__ import annotations
import io, os, shutil, subprocess, sys, tempfile
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
BN = os.path.join("src", "kernel_optimizer", "evaluation", "bottleneck.py")
TESTS = ["tests/test_zero_is_a_measurement.py"]
VARIANTS = [
    ("the_original_falsy_drop", BN,
     ["    if n_spills is not None:", "        ev[\"n_spills\"] = n_spills"],
     ["    if n_spills:", "        ev[\"n_spills\"] = n_spills"],
     "a measured ZERO is recorded as unmeasured -- 11 of 128 records on box 2"),
    ("n_regs_falsy_drop", BN,
     ["    if n_regs is not None:", "        ev[\"n_regs\"] = n_regs"],
     ["    if n_regs:", "        ev[\"n_regs\"] = n_regs"],
     "same bug on the neighbouring dimension"),
    ("near_limit_also_keeps_zero", BN,
     ["    if n_spills:", "        near_limit.append(f\"spills={n_spills}\")"],
     ["    if n_spills is not None:", "        near_limit.append(f\"spills={n_spills}\")"],
     "the over-correction: `spills=0` then appears in at_limit for every clean candidate"),
]
def run(cwd):
    env = dict(os.environ, PYTHONPATH=os.path.join(cwd, "src"))
    p = subprocess.Popen([sys.executable, "-m", "pytest"] + TESTS + ["-q"], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o, _ = p.communicate()
    return p.returncode, o.decode("utf-8", "replace")
base = tempfile.mkdtemp(prefix="rc-zero-")
work = os.path.join(base, "opop")
shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "runs",
                                                          "runs-*", ".venv", "sandboxes"))
rc, out = run(work)
if rc != 0:
    print("BASELINE FAILS:"); print(out[-2000:]); raise SystemExit(2)
print("baseline: %s" % out.strip().splitlines()[-1])
bad = 0
for name, rel, old_l, new_l, why in VARIANTS:
    path = os.path.join(work, rel)
    orig = io.open(path, encoding="utf-8").read()
    old, new = NL.join(old_l), NL.join(new_l)
    if old not in orig:
        print("  %-34s INVALID -- anchor absent" % name); bad += 1; continue
    patched = orig.replace(old, new, 1)
    if patched == orig:
        print("  %-34s INVALID -- no bytes changed" % name); bad += 1; continue
    io.open(path, "w", encoding="utf-8", newline=NL).write(patched)
    try:
        rc, out = run(work)
    finally:
        io.open(path, "w", encoding="utf-8", newline=NL).write(orig)
    tail = out.strip().splitlines()[-1] if out.strip() else "(none)"
    if rc == 0:
        print("  %-34s **NOT CAUGHT** -- %s" % (name, why)); print("      %s" % tail); bad += 1
    else:
        print("  %-34s CAUGHT  (%s)" % (name, tail))
shutil.rmtree(base, ignore_errors=True)
print(); print("%d not caught or invalid" % bad)
raise SystemExit(1 if bad else 0)
