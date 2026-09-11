#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Revert-check for the prescreen's own timeout. Every variant must be CAUGHT.

The rule this enforces (recorded twice in this project): a test suite that passes on the code is no
evidence at all until each wrong implementation it names is shown to FAIL it. And two prior instances
of the meta-failure are guarded against here directly:

  * `a-variant-that-changes-no-behaviour-is-not-a-variant` -- each patch is checked for having
    actually changed the file, and a variant whose behaviour is unchanged is reported as INVALID
    rather than counted as CAUGHT.
  * a non-applying edit reading as a pass -- the anchor is asserted present BEFORE the patch and the
    result is asserted different AFTER, so a mangled anchor is a loud failure.

    python revert_check_prescreen_timeout.py [<repo_root>]
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = os.path.join("src", "kernel_optimizer", "evaluation", "correctness.py")
ORCH = os.path.join("src", "kernel_optimizer", "control", "orchestrator.py")
TESTS = ["tests/test_prescreen_timeout.py", "tests/test_prescreen_observability.py",
         # The safety argument -- "a timeout caches nothing, so no configuration is
         # excluded" -- is what makes a short prescreen deadline a cost cap rather than a
         # search-space restriction. Its guard lives in test_improvements
         # (`test_a_failed_probe_is_never_cached_as_a_screen_verdict`, from the box 3
         # SIGTERM case), so the variant that breaks it must be run against THAT suite too
         # or it reads as NOT CAUGHT here while being caught in the repo.
         "tests/test_improvements.py::test_a_failed_probe_is_never_cached_as_a_screen_verdict"]

# (name, file, old, new, why this is a plausible wrong implementation)
_NL = chr(10)

VARIANTS = [
    ("the_original_defect_build_timeout_s", CORR,
     "        deadline = prescreen_timeout_s(self.cfg, len(pending))",
     "        deadline = self.cfg.build_timeout_s",
     "the state before the fix: the screen borrows the real trial's 1200 s compile budget"),
    ("deadline_computed_inside_the_swallowing_try", CORR,
     _NL.join(['        deadline = prescreen_timeout_s(self.cfg, len(pending))', '        try:', '            probe = self.worker.run_job(job, deadline, f"{tag}-prescreen", lock_mode="shared")']),
     _NL.join(['        try:', '            probe = self.worker.run_job(job, prescreen_timeout_s(self.cfg, len(pending)), f"{tag}-prescreen", lock_mode="shared")']),
     "a config missing the prescreen fields then raises inside `except Exception`, is "
     "swallowed as {ok: False} and reads as a PROBE failure -- so a mis-wired config is "
     "indistinguishable from a timing-out ptxas. The defect the FIRST version of this "
     "change actually had, found by test_a_failed_probe_is_never_cached_as_a_screen_verdict"),
    ("no_clamp_so_the_screen_can_outlast_a_trial", CORR,
     "    return min(float(budget), float(cfg.build_timeout_s))",
     "    return float(budget)",
     "a mis-set config then makes the SCREEN the longer deadline, restoring the tail silently"),
    ("a_flat_constant_that_ignores_batch_size", CORR,
     "    budget = (cfg.prescreen_base_timeout_s\n"
     "              + cfg.prescreen_per_config_timeout_s * max(0, n_configs))",
     "    budget = cfg.prescreen_base_timeout_s",
     "one constant: too tight for a 40-config batch, and D5 measured 76-260 s for 40"),
    ("a_timeout_is_cached_so_configs_are_excluded", CORR,
     "            if not entry.get(\"ok\"):",
     "            if False and not entry.get(\"ok\"):",
     "caching a non-answer makes the shorter deadline REMOVE configurations -- the exact thing "
     "that would turn a cost cap into a search-space restriction"),
    ("event_reads_build_timeout_s_instead", ORCH,
     "        limit = prescreen_timeout_s(self.cfg.evaluation, len(paths))",
     "        limit = self.cfg.evaluation.build_timeout_s",
     "then `timed_out` reads False on a screen that WAS cut off -- the silent direction"),
]


def run(cwd: str) -> tuple[int, str]:
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=os.path.join(cwd, "src"))
    p = subprocess.Popen([py, "-m", "pytest"] + TESTS + ["-q"], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate()
    return p.returncode, out.decode("utf-8", "replace")


def main() -> int:
    base = tempfile.mkdtemp(prefix="revert-prescreen-")
    work = os.path.join(base, "opop")
    shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "runs", "runs-*", ".venv", "sandboxes"))

    rc, out = run(work)
    if rc != 0:
        print("BASELINE FAILS -- the suite must be green before any variant means anything:\n%s"
              % out[-3000:])
        return 2
    print("baseline: %s" % out.strip().splitlines()[-1])

    bad = 0
    for name, rel, old, new, why in VARIANTS:
        path = os.path.join(work, rel)
        orig = io.open(path, encoding="utf-8").read()
        if old not in orig:
            print("  %-46s INVALID -- anchor absent, so nothing was patched" % name)
            bad += 1
            continue
        patched = orig.replace(old, new, 1)
        if patched == orig:
            print("  %-46s INVALID -- the patch changed no bytes" % name)
            bad += 1
            continue
        io.open(path, "w", encoding="utf-8", newline="\n").write(patched)
        try:
            rc, out = run(work)
        finally:
            io.open(path, "w", encoding="utf-8", newline="\n").write(orig)
        tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        if rc == 0:
            print("  %-46s **NOT CAUGHT** -- %s\n      %s" % (name, why, tail))
            bad += 1
        else:
            print("  %-46s CAUGHT  (%s)" % (name, tail))

    shutil.rmtree(base, ignore_errors=True)
    print("\n%d variant(s) not caught or invalid" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
