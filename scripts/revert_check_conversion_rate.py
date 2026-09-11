# -*- coding: utf-8 -*-
"""Revert-check the per-candidate conversion RATE. Every variant must be CAUGHT.

Both directions, and the second matters as much as the first:

  * the rate stops being estimated, or is estimated on evidence that does not support it -- a
    confidently-wrong slope is worse than none, because a consumer treats it as a prior;
  * the rate starts being CONSUMED -- which silently ends the comparability that makes it safe to
    land this before E1/E3.

Anchors are explicit line lists joined with chr(10); never a shell heredoc, which rewrites escapes.
"""
from __future__ import annotations
import io, os, shutil, subprocess, sys, tempfile

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
CR = os.path.join("src", "kernel_optimizer", "evaluation", "conversion_rate.py")
ORCH = os.path.join("src", "kernel_optimizer", "control", "orchestrator.py")
TESTS = ["tests/test_conversion_rate.py"]

VARIANTS = [
    # ---- the evidence gates stop holding ----------------------------------------------
    ("two_points_are_enough", CR,
     ["MIN_DISTINCT = 3"],
     ["MIN_DISTINCT = 2"],
     "two points define a line through themselves and report r2=1.0 -- the most confident-looking "
     "output possible and the least informative"),

    ("no_r2_gate", CR,
     ["MIN_R2 = 0.4"],
     ["MIN_R2 = 0.0"],
     "a dimension with no relationship gets a slope; n_regs/occupancy sit at r2 0.03-0.08 across "
     "the corpus and would all be reported"),

    ("four_trials_are_enough", CR,
     ["MIN_TRIALS = 8"],
     ["MIN_TRIALS = 4"],
     "a slope from 4 trials is a curiosity, not a measurement"),

    ("regs_added_to_the_rated_dimensions", CR,
     ['RATE_DIMENSIONS = ("n_spills",)'],
     ['RATE_DIMENSIONS = ("n_spills", "n_regs")'],
     "n_regs' slope sign flips between candidates (13+/30- on box 1, 15+/36- on box 2), so a rate "
     "for it would be wrong in the sign for most candidates"),

    ("sign_reversal_check_removed", CR,
     ["        if note:", "            continue"],
     ["        if False:", "            continue"],
     "a slope that reverses once the zeros are removed is a jump between two regimes, not a rate "
     "inside either -- the 2-of-51 case on box 2"),

    ("a_measured_zero_is_dropped", CR,
     ["        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):"],
     ["        if not v or isinstance(v, bool) or not isinstance(v, (int, float)):"],
     "truthiness deletes the clean end of every series; the same falsy drop one layer up cost 11 "
     "of 128 dimension records on box 2"),

    ("the_mean_is_used_instead_of_the_median", CR,
     ['        ms = getattr(lat, "robust_ms", None) if lat is not None else None'],
     ['        ms = getattr(lat, "mean", None) if lat is not None else None'],
     "the fit then relates the resource to a statistic no decision used; the mean's ranking "
     "accuracy on this hardware is 64.8% against the median's 93.2%"),

    ("failed_trials_enter_the_fit", CR,
     ['        if getattr(t, "status", None) != "complete":', "            continue"],
     ["        if False:", "            continue"],
     "a failed trial has no latency and would be read as a data point"),

    # ---- the journalling stops being honest -------------------------------------------
    ("nothing_is_journalled_when_no_rate_clears", CR,
     ["    rates = conversion_rates(trials)", "    return {"],
     ["    rates = conversion_rates(trials)", "    if not rates:", "        return {}", "    return {"],
     "an empty payload reads exactly like a mechanism that never ran -- the shape that left "
     "launch_bound unreachable across 848 trials"),

    ("the_event_is_never_emitted", ORCH,
     ['            self.store.append("CONVERSION_RATES", {'],
     ['            _unused = ("CONVERSION_RATES", {'],
     "the estimator runs and its output goes nowhere"),

    ("the_estimator_failure_is_swallowed_silently", ORCH,
     ['            self.store.append("CONVERSION_RATES_FAILED", {'],
     ['            _swallowed = ("CONVERSION_RATES_FAILED", {'],
     "a diagnostic that fails invisibly is indistinguishable from one that found nothing"),

    # ---- the comparability guarantee ends ---------------------------------------------
    ("the_rate_starts_being_consumed", ORCH,
     ['            from kernel_optimizer.evaluation.conversion_rate import rates_payload'],
     ['            from kernel_optimizer.evaluation.conversion_rate import (',
      '                conversion_rates, rates_payload)',
      '            _slopes = [r.slope_ms_per_unit for r in conversion_rates(crun.trials)]'],
     "step 1 must journal the number and NOT consume it: a run that ranked, allocated or prompted "
     "differently because of it is no longer comparable with the three completed runs or with "
     "either E1 arm, which is the whole reason it is safe to land this now"),
]


def run(cwd):
    env = dict(os.environ, PYTHONPATH=os.path.join(cwd, "src"))
    p = subprocess.Popen([sys.executable, "-m", "pytest"] + TESTS + ["-q"], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o, _ = p.communicate()
    return p.returncode, o.decode("utf-8", "replace")


base = tempfile.mkdtemp(prefix="rc-rate-")
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
        print("  %-46s INVALID -- anchor absent" % name); bad += 1; continue
    patched = orig.replace(old, new, 1)
    if patched == orig:
        print("  %-46s INVALID -- no bytes changed" % name); bad += 1; continue
    io.open(path, "w", encoding="utf-8", newline=NL).write(patched)
    try:
        rc, out = run(work)
    finally:
        io.open(path, "w", encoding="utf-8", newline=NL).write(orig)
    tail = out.strip().splitlines()[-1] if out.strip() else "(none)"
    if rc == 0:
        print("  %-46s **NOT CAUGHT** -- %s" % (name, why)); print("      %s" % tail); bad += 1
    else:
        print("  %-46s CAUGHT  (%s)" % (name, tail))
shutil.rmtree(base, ignore_errors=True)
print()
print("%d not caught or invalid" % bad)
raise SystemExit(1 if bad else 0)
