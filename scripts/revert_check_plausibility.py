# -*- coding: utf-8 -*-
"""Revert-check the A1 derived plausibility bound. Every variant must be CAUGHT.

The variants cover BOTH directions, because a bound has two failure modes and only testing one
is how the 10x constant survived 25 runs:

  * the bound stops flagging  -> a work-skipping kernel is reported as a spectacular win;
  * the bound flags honestly-fast kernels -> exactly the defect being removed, where L3:48's
    verified-correct 14.29x winner cost a manual re-verification on all three runs.

Anchors are built from explicit line lists joined with chr(10) and never written through a shell
heredoc: a heredoc rewrites Python escapes (a "\\n" inside one becomes a real newline), and a
patch that does not apply has twice been read as a pass in this project.
"""
from __future__ import annotations
import io, os, shutil, subprocess, sys, tempfile

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
PL = os.path.join("src", "kernel_optimizer", "evaluation", "plausibility.py")
ORCH = os.path.join("src", "kernel_optimizer", "control", "orchestrator.py")
CORR = os.path.join("src", "kernel_optimizer", "evaluation", "correctness.py")
JOBS = os.path.join("src", "kernel_optimizer", "gpu", "jobs.py")
TESTS = ["tests/test_plausibility_bound.py", "tests/test_plausibility_wiring.py",
         "tests/test_improvements.py"]

VARIANTS = [
    # ---- the bound stops binding -------------------------------------------------------
    ("floor_uses_fp32_peak_only", PL,
     ['    names = ("fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops",'],
     ['    names = ("fp32_tflops",  # noqa'],
     "the arithmetic floor is computed from fp32 while candidates legally use tensor cores: "
     "12x too high on the A800, so every low-precision kernel is flagged"),

    ("no_margin_at_all", PL,
     ["DEFAULT_MARGIN = 1.5"],
     ["DEFAULT_MARGIN = 1.0"],
     "the bare physical ceiling flags a kernel that merely beats a streaming benchmark's "
     "measured bandwidth, which a better access pattern legitimately can"),

    ("l2_test_removed_so_a_cached_workload_gets_a_bogus_dram_floor", PL,
     ["        if l2_bytes > 0 and compulsory_bytes <= l2_bytes:",
      "            dram_applicable = False"],
     ["        if False:",
      "            dram_applicable = False"],
     "a logical byte count is treated as bus traffic for an L2-resident task: the floor "
     "collapses to microseconds and nothing is ever flagged"),

    ("unmeasured_l2_treated_as_fitting", PL,
     ["        if l2_bytes > 0 and compulsory_bytes <= l2_bytes:"],
     ["        if compulsory_bytes <= l2_bytes or l2_bytes == 0:"],
     "l2_bytes=0 means UNMEASURED; reading it as 'fits' disables the DRAM floor on every box "
     "whose calibration predates the L2 probe -- which includes the A800's cached one"),

    ("zero_floor_returns_a_bound_instead_of_None", PL,
     ["    if floor_ms <= 0:", "        # Neither term measurable."],
     ["    if False:", "        # Neither term measurable."],
     "a zero floor yields an infinite ceiling, so every speedup is 'possible' forever"),

    ("no_calibration_falls_back_to_the_10x_constant", ORCH,
     ["        setter(ceiling)"],
     ["        setter(ceiling)  # noqa" + NL +
      "        if ceiling is None:" + NL +
      "            from kernel_optimizer.evaluation.plausibility import SpeedupCeiling" + NL +
      "            setter(SpeedupCeiling(threshold_x=10.0, ceiling_x=10.0, floor_ms=1.0," + NL +
      "                                  margin=1.0, binding_term='dram', reference_ms=10.0," + NL +
      "                                  derivation='legacy 10x'))"],
     "the removed constant is reinstated on exactly the boxes that cannot derive a bound"),

    # ---- the bound flags honest kernels ------------------------------------------------
    ("reference_is_torch_compile_not_eager", ORCH,
     ['                    if b.kind.startswith("eager") and b.latency_ms.mean > 0]'],
     ['                    if b.latency_ms.mean > 0]'],
     "torch.compile enters the numerator and shrinks L3:48's ceiling from 17.48x to 10.7x, "
     "re-flagging the very verified-correct winner this change stops flagging"),

    ("reference_is_the_fastest_eager_not_the_slowest", ORCH,
     ["                reference_ms=max(eager_ms),"],
     ["                reference_ms=min(eager_ms),"],
     "an error in the numerator now NARROWS the bound instead of widening it, so uncertainty "
     "produces false flags rather than silence"),

    ("compute_term_dropped_so_an_l2_task_has_no_bound_left", PL,
     ["    if flop_count > 0 and peak_tflops > 0:",
      "        compute_floor_ms = (flop_count / (peak_tflops * 1e12)) * 1e3"],
     ["    if False:",
      "        compute_floor_ms = (flop_count / (peak_tflops * 1e12)) * 1e3"],
     "with no arithmetic term an L2-resident task loses its only remaining floor"),

    # ---- the wiring stops delivering ---------------------------------------------------
    ("bound_is_derived_but_never_installed", ORCH,
     ["        setter(ceiling)"],
     ["        pass  # setter(ceiling)"],
     "the bound is computed, journalled and never reaches a single job: the log looks correct "
     "while nothing is checked -- the shape that left launch_bound unreachable for 848 trials"),

    ("evaluator_drops_the_fields_on_the_way_to_the_job", CORR,
     ['        return dict(self._plausibility) if self._plausibility else None'],
     ['        return None'],
     "installed and then silently withheld from every job"),

    ("job_builder_ignores_the_plausibility_argument", JOBS,
     ["    if plausibility:", "        job.update(plausibility)", "    return job"],
     ["    if False:", "        job.update(plausibility)", "    return job"],
     "the field never reaches the worker, so no timed trial is ever checked"),

    ("worker_defaults_a_missing_threshold_back_to_10x",
     os.path.join("src", "kernel_optimizer", "gpu", "worker_main.py"),
     ["    if thr is None:", "        return None"],
     ["    if thr is None:", "        return 10.0"],
     "a job carrying no derived bound silently reverts to the constant"),
]


def run(cwd):
    env = dict(os.environ, PYTHONPATH=os.path.join(cwd, "src"))
    p = subprocess.Popen([sys.executable, "-m", "pytest"] + TESTS + ["-q"], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o, _ = p.communicate()
    return p.returncode, o.decode("utf-8", "replace")


base = tempfile.mkdtemp(prefix="rc-plaus-")
work = os.path.join(base, "opop")
shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "runs",
                                                          "runs-*", ".venv", "sandboxes"))
rc, out = run(work)
if rc != 0:
    print("BASELINE FAILS:"); print(out[-3000:]); raise SystemExit(2)
print("baseline: %s" % out.strip().splitlines()[-1])
bad = 0
for name, rel, old_l, new_l, why in VARIANTS:
    path = os.path.join(work, rel)
    orig = io.open(path, encoding="utf-8").read()
    old, new = NL.join(old_l), NL.join(new_l)
    if old not in orig:
        print("  %-52s INVALID -- anchor absent" % name); bad += 1; continue
    patched = orig.replace(old, new, 1)
    if patched == orig:
        print("  %-52s INVALID -- no bytes changed" % name); bad += 1; continue
    io.open(path, "w", encoding="utf-8", newline=NL).write(patched)
    try:
        rc, out = run(work)
    finally:
        io.open(path, "w", encoding="utf-8", newline=NL).write(orig)
    tail = out.strip().splitlines()[-1] if out.strip() else "(none)"
    if rc == 0:
        print("  %-52s **NOT CAUGHT** -- %s" % (name, why)); print("      %s" % tail); bad += 1
    else:
        print("  %-52s CAUGHT  (%s)" % (name, tail))
shutil.rmtree(base, ignore_errors=True)
print()
print("%d not caught or invalid" % bad)
raise SystemExit(1 if bad else 0)
