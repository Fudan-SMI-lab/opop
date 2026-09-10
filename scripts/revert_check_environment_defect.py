"""Revert-check for the environment-defect diagnosis: every test must FAIL on the wrong version.

The fix this guards is small and its failure mode is silence, which is the combination that most
needs a failing control. Two directions matter equally and they pull against each other:

  TOO NARROW -- the diagnosis never fires, and a box defect reaches the repair agent as "your kernel
  crashed". Measured cost: 12 of 12 candidates `runtime_error` from one missing dependency, read as
  "the model wrote bad kernels" (G29), then the same failure again on a fresh venv.

  TOO BROAD -- a candidate's own missing import is reclassified as a box defect, so the repair loop
  stops seeing real failures. That trades a diagnosis problem for a lost-samples problem, which is
  worse: an agent reaching for CUTLASS or TileLang is a real case that must keep arriving as the
  candidate's problem.

So the variants below deliberately include BOTH over- and under-matching, and the suite has to
discriminate each.

Same contract and hazards as `revert_check_s2.py` (see its `run()` docstring for why `-B` and `-v`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEN = ROOT / "src" / "kernel_optimizer" / "evaluation" / "benchmark.py"
ORC = ROOT / "src" / "kernel_optimizer" / "control" / "orchestrator.py"
TESTS = [ROOT / "tests" / "test_environment_defect.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "the pre-fix state: no diagnosis at all",
        BEN,
        '    if "ModuleNotFoundError" not in log_tail and "ImportError" not in log_tail:\n'
        "        return None",
        "    return None\n"
        '    if "ModuleNotFoundError" not in log_tail and "ImportError" not in log_tail:\n'
        "        return None",
        ["test_the_real_box1_failure_is_recognised",
         "test_the_g29_static_check_failure_is_recognised_too",
         "test_the_message_names_wsl_venv_and_not_the_launching_interpreter",
         "test_the_baseline_raise_carries_the_diagnosis"],
        "the state that cost a run today and 12 candidates before that: a missing dependency and a "
        "wrong kernel are indistinguishable, both arriving as `runtime_error` plus a traceback",
    ),
    (
        "TOO BROAD: any import failure counts as a box defect",
        BEN,
        '    if "kernelbench/__init__" not in log_tail and "kernelbench/utils" not in log_tail:\n'
        "        return None",
        '    if False and "kernelbench/__init__" not in log_tail:\n'
        "        return None",
        ["test_a_candidate_importing_an_absent_module_is_NOT_an_environment_defect",
         "test_a_harness_module_import_failure_inside_the_worker_is_not_matched_either",
         "test_an_import_error_outside_the_kernelbench_chain_is_not_matched"],
        "the WORSE failure direction. A candidate reaching for CUTLASS or TileLang -- neither "
        "installed, and a measured real case -- is relabelled a box defect, so the repair loop stops "
        "seeing it. It also claims G33's harness-import degradation, whose fix is a different knob "
        "(extra_pythonpath), so the message would point at the wrong repair",
    ),
    (
        "TOO BROAD the other way: any kernelbench frame counts, import failure or not",
        BEN,
        '    if "ModuleNotFoundError" not in log_tail and "ImportError" not in log_tail:\n'
        "        return None",
        '    if False and "ModuleNotFoundError" not in log_tail:\n'
        "        return None",
        ["test_the_two_signals_are_independent_not_one_regex",
         "test_ordinary_failures_are_not_environment_defects[oom-through-kernelbench-utils-frame]"],
        "the two signals collapse to one. Any traceback passing through kernelbench -- which is "
        "EVERY correctness failure, since kernelbench is the evaluator -- acquires a 'fix your box' "
        "message. An operator learns to distrust it, and a warning nobody trusts is not a warning. "
        "NOTE this variant EXPOSED A REAL GAP in my tests: the original `ordinary_failures` cases "
        "were bare strings with no kernelbench frame, so every one of them survived a "
        "frame-only matcher CORRECTLY and the variant looked like it did not discriminate. Two cases "
        "carrying real kernelbench frames were added because of it -- found by the harness, not by "
        "inspection. And only the `utils.py` case may be NAMED: the `eval.py` one carries neither "
        "watched frame, so it correctly survives a frame-only matcher -- it is evidence that "
        "the matcher is anchored on the IMPORT CHAIN rather than on kernelbench generally, "
        "not evidence about this variant",
    ),
    (
        "the missing module name dropped from the message",
        BEN,
        "    missing = \"\"\n"
        "    for line in log_tail.splitlines():",
        "    missing = \"\"\n"
        "    for line in []:",
        ["test_the_real_box1_failure_is_recognised",
         "test_the_g29_static_check_failure_is_recognised_too"],
        "'something in the import chain failed' is not actionable. The operator has to reproduce the "
        "failure to learn which package to install -- and the chain is four deep (dotenv -> openai "
        "-> litellm -> tiktoken), so it is four round trips instead of one",
    ),
    (
        "the message points at the launching interpreter instead of wsl.venv",
        BEN,
        '"venv named by `wsl.venv` in this run\'s config (NOT the interpreter that launched the CLI "',
        '"venv (the one you launched this from) "',
        ["test_the_message_names_wsl_venv_and_not_the_launching_interpreter"],
        "sends the operator to the WRONG interpreter, where the install changes nothing and the "
        "failure repeats. Not hypothetical: box 2 deliberately runs the driver from orch-venv and "
        "the worker from kernel-opt-venv, so 'the venv you launched from' is the one that does not "
        "matter",
    ),
    (
        "the fatal baseline path computes the diagnosis and drops it",
        BEN,
        '                            + (f"\\n\\n{env}" if env else "")',
        "",
        ["test_the_baseline_raise_carries_the_diagnosis"],
        "the `conversion` defect restated (G44): computed, correct, and read by nothing. The "
        "baseline failure is fatal by design, so the raised message is ALL the operator gets",
    ),
    (
        "the diagnosis appended unconditionally",
        BEN,
        '                        env = environment_defect(tail)',
        '                        env = environment_defect(tail) or "ENVIRONMENT DEFECT: check the box"',
        ["test_the_baseline_raise_stays_bare_for_an_ordinary_failure"],
        "an OOM or a correctness miss now tells the operator to fix their box. This variant exists "
        "because the positive test alone is satisfied by appending the text always -- which is why "
        "the negative-direction test is not optional",
    ),
    (
        "the per-candidate path left unannotated (the pre-fix state there)",
        ORC,
        "            env = environment_defect(tail)\n"
        "            if env:\n"
        "                detail = f\"{detail}\\n\\n{env}\"",
        "            pass",
        ["test_the_candidate_trial_path_carries_the_diagnosis"],
        "the path that matters MORE than the baseline: the baseline kills the run loudly, while a "
        "per-trial failure is silent and gets attributed to the model. This is exactly where G29's "
        "12 candidates were lost. THIS VARIANT CAUGHT A DEFECT IN MY OWN TEST: the test used to "
        "assert on `inspect.getsource(_run_trial)` and PASSED with the whole block replaced by "
        "`pass`, because `failure_detail=detail` remained on the TrialRecord below the deleted "
        "lines. A source-text assertion passing on broken code, which this project has a recorded "
        "failure mode for. It is now behavioural -- it drives the real `_run_trial` with a faked "
        "evaluator and reads the TrialRecord. Note it only surfaced on the A800: on Windows the "
        "test SKIPS for lack of optuna, so the harness reported UNVERIFIED, not FAIL. A skip is "
        "not a pass, and it is not a verdict either",
    ),
    (
        "TOO BROAD on the trial path: the diagnosis appended to every failure",
        ORC,
        "            env = environment_defect(tail)\n"
        "            if env:\n"
        "                detail = f\"{detail}\\n\\n{env}\"",
        "            env = environment_defect(tail) or \"ENVIRONMENT DEFECT: check the box\"\n"
        "            detail = f\"{detail}\\n\\n{env}\"",
        ["test_the_candidate_trial_path_stays_bare_for_an_ordinary_failure"],
        "the counter-direction on the trial path, which the baseline path already had but this one "
        "did not. Without it the positive test above is satisfied by appending the text "
        "unconditionally -- and then every correctness miss and every OOM tells the operator to fix "
        "their box. A warning that fires on everything is not a warning, and here it would also "
        "mislabel a candidate's own missing import (CUTLASS, TileLang) as the box's fault",
    ),
]


def run(names: list[str]) -> tuple[set[str], set[str], bool, str]:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", *[str(t) for t in TESTS], "-v", "--no-header",
         "--tb=no", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    out = proc.stdout + proc.stderr
    failed: set[str] = set()
    skipped: set[str] = set()
    for line in out.splitlines():
        for n in names:
            if f"::{n} " not in line:
                continue
            if "FAILED" in line or "ERROR" in line:
                failed.add(n)
            elif "SKIPPED" in line:
                skipped.add(n)
    return failed, skipped, proc.returncode == 0, out


def apply_variant(path: Path, text: str, old: str, new: str, names: list[str]):
    path.write_text(text.replace(old, new), encoding="utf-8")
    try:
        failed, skipped, _, out = run(names)
    finally:
        path.write_text(text, encoding="utf-8")
    return failed, skipped, out


def main() -> int:
    originals = {p: p.read_text(encoding="utf-8") for p in (BEN, ORC)}
    ok = True
    unverified = 0
    unstable: list[str] = []
    try:
        _, _, green, out = run([])
        if not green:
            print("BASELINE IS NOT GREEN -- nothing below means anything\n" + out[-3000:])
            return 2
        print("baseline: green (%s)\n" % out.strip().splitlines()[-1])
        if "SKIPPED" in out:
            print("!! Some tests SKIP here (no optuna). A skip is not a failure, so a variant whose")
            print("!! named tests all skip is UNVERIFIED on this box -- re-run on the A800.\n")

        for label, path, old, new, must_fail, why in VARIANTS:
            text = originals[path]
            if text.count(old) != 1:
                print("**SKIPPED** %s: anchor occurs %d times, not once -- the variant would not be "
                      "the change it claims to be" % (label, text.count(old)))
                ok = False
                continue

            patched = text.replace(old, new)
            try:
                compile(patched, str(path), "exec")
            except SyntaxError as exc:
                print("**SKIPPED** %s: the patched file does not parse (%s), so any failure it "
                      "produced would be for the wrong reason" % (label, exc))
                ok = False
                continue

            failed, was_skipped, out = apply_variant(path, text, old, new, must_fail)
            missing = [n for n in must_fail if n not in failed]
            unrun = [n for n in missing if n in was_skipped]
            wrongly_passed = [n for n in missing if n not in was_skipped]

            if wrongly_passed:
                failed2, _, _ = apply_variant(path, text, old, new, must_fail)
                still = [n for n in wrongly_passed if n not in failed2]
                if len(still) != len(wrongly_passed):
                    unstable.append(label)
                    ok = False
                    print("**UNSTABLE** %s" % label)
                    print("        two identical runs DISAGREED; neither answer can be reported")
                    print("        wrong version: %s\n" % why)
                    continue
                ok = False
                print("**FAIL** %s" % label)
                print("        these tests PASSED on the wrong implementation, so they are not")
                print("        evidence for it: %s" % ", ".join(wrongly_passed))
            elif unrun and len(unrun) == len(must_fail):
                unverified += 1
                print("UNVERIF %s" % label)
                print("        every named test SKIPPED here: %s" % ", ".join(unrun))
            else:
                verified = [n for n in must_fail if n in failed]
                print("ok      %s" % label)
                print("        %d/%d named tests failed as required%s" % (
                    len(verified), len(must_fail),
                    "" if not unrun else " (%d skipped here)" % len(unrun)))
            print("        wrong version: %s" % why)
            print()
    finally:
        for p, text in originals.items():
            p.write_text(text, encoding="utf-8")

    _, _, green, out = run([])
    if not green:
        print("!! RESTORE DID NOT COME BACK GREEN -- check git status\n" + out[-2000:])
        return 2
    print("restored, suite green again")
    if unstable:
        print("\nVERDICT: harness UNSTABLE on %d variant(s); no verdict is trustworthy: %s"
              % (len(unstable), ", ".join(unstable)))
        return 1
    if not ok:
        print("\nVERDICT: at least one test is not evidence -- see **FAIL** above")
        return 1
    if unverified:
        print("\nVERDICT: every variant this box could check discriminates; %d UNVERIFIED here"
              % unverified)
        return 0
    print("\nVERDICT: every test fails on the implementation it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
