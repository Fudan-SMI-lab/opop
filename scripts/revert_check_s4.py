"""Revert-check for S4': every test must FAIL on the wrong implementation it names.

Same contract and hazards as `revert_check_s2.py` -- see that file's `run()` docstring for why `-B` is
required and why outcomes are read from a line that names the test.

The variant list leads with the one that matters most: computing the section and not calling it. That
IS the gap S4' closes -- `conversion_verdict` was correct, journalled, and read by nothing -- and it is
invisible to every test that drives the helper directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CR = ROOT / "src" / "kernel_optimizer" / "evaluation" / "conversion_report.py"
REP = ROOT / "src" / "kernel_optimizer" / "reporting" / "report.py"
TESTS = [ROOT / "tests" / "test_s4_conversion_consumer.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "the section is computed and never consumed (the gap itself)",
        REP,
        """        lines.extend(conversion_lines(
            events,
            min_improvement_pct=float((budgets or {}).get("min_improvement_pct", 2.0))))""",
        """        _unused_s4 = conversion_lines(
            events,
            min_improvement_pct=float((budgets or {}).get("min_improvement_pct", 2.0)))""",
        ["test_the_section_reaches_the_GENERATED_report_not_only_the_helper"],
        "exactly the pre-S4' state: the verdict is computed, journalled, and READ BY NOTHING. Every "
        "test that drives `conversion_lines` directly still passes, which is why the end-to-end test "
        "had to exist",
    ),
    (
        "complementary slackness always passes",
        CR,
        "    out: list[SlacknessViolation] = []\n    floor = min_improvement_pct * _SLACK_TOLERANCE_MULTIPLE",
        "    return []\n    out: list[SlacknessViolation] = []\n    floor = min_improvement_pct * _SLACK_TOLERANCE_MULTIPLE",
        ["test_a_slack_dimension_that_bought_speed_is_reported_as_our_error",
         "test_a_gain_below_the_runs_own_threshold_is_not_a_violation",
         "test_the_violation_carries_its_attribution_caveat",
         "test_the_slack_set_is_a_union_and_the_direction_of_that_error_is_toward_noise",
         "test_the_section_reaches_the_GENERATED_report_not_only_the_helper"],
        "the check that grades OUR OWN verdicts becomes a silent pass forever -- the shape of a probe "
        "with no positive control, and this project has already read five negative 'results' off a "
        "broken probe",
    ),
    (
        "the theorem applied in both directions",
        CR,
        "        if not isinstance(gain, (int, float)) or gain < floor:\n            continue",
        "        if not isinstance(gain, (int, float)):\n            continue",
        ["test_a_slack_dimension_that_bought_nothing_is_the_theorem_holding_and_is_not_reported",
         "test_a_gain_below_the_runs_own_threshold_is_not_a_violation"],
        "a slack dimension improving with NO latency gain is the theorem HOLDING; reporting it buries "
        "the real violations in noise, and `no_conversion` on a slack dimension is exactly what the "
        "theorem predicts",
    ),
    (
        "a local noise-floor constant instead of the run's own threshold",
        CR,
        "    floor = min_improvement_pct * _SLACK_TOLERANCE_MULTIPLE",
        "    floor = 0.5",
        ["test_a_gain_below_the_runs_own_threshold_is_not_a_violation"],
        "a second opinion about the same quantity. The noise floor is a property of (card, task) -- on "
        "L3:48 the per-trial std was 16% of the mean while 33 near-ties spanned 9% -- and the run "
        "already carries a threshold derived for it",
    ),
    (
        "a binding dimension counted as a violation",
        CR,
        "            if dim not in slack_dimensions:\n                continue",
        "            if False:\n                continue",
        ["test_a_binding_dimension_that_bought_speed_is_not_a_violation"],
        "fires on every improvement, including a binding dimension buying speed -- which is the system "
        "WORKING. A check that always fires is as useless as one that never does",
    ),
    (
        "an empty slack set reported as a pass",
        CR,
        "    elif not slack_dimensions:",
        "    elif False:",
        ["test_no_slack_dimensions_means_the_check_cannot_run_and_says_so_rather_than_passing"],
        "'nothing was slack' is reported as 'no violation', i.e. a check with nothing to test reports "
        "success -- the exact reading that turned a broken probe into five negative results",
    ),
    (
        "the clean pass presented as strong",
        CR,
        '                "This is a WEAK pass. The slack set is a union over the whole run (see "',
        '                "" or "This is a strong pass. The slack set is a union over the whole run (see "',
        ["test_a_clean_pass_is_labelled_weak_and_says_why"],
        "a weak pass gets cited as a strong one. It passes vacuously with few rounds, and the slack "
        "set is a union over the run",
    ),
    (
        "the attribution caveat dropped from a violation",
        CR,
        '        out.append("Attribution caveat: a rewrite is RE-TUNED, so the gain may come from elsewhere "\n'
        '                   "in the same round. One coincidence is a question; a repeated one is a verdict to "\n'
        '                   "distrust.\\n")',
        '        out.append("(no caveat)\\n")',
        ["test_the_violation_carries_its_attribution_caveat"],
        "a single coincidence reads as a refutation of our verdict, when a rewrite is re-tuned and the "
        "gain may come from elsewhere in the same round",
    ),
    (
        "rounds without a conversion field rendered as an empty table",
        CR,
        "    if not with_verdict:",
        "    if False:",
        ["test_rounds_without_a_conversion_field_are_reported_as_predating_the_fix"],
        "the reader cannot tell 'nothing converted' from 'the mechanism never ran' -- and the second is "
        "the state of ALL FIVE existing corpora (9 of 9 rounds with no conversion field)",
    ),
    (
        "a run with no rewrite rounds renders nothing at all",
        CR,
        "    if not rounds:",
        "    if False and not rounds:",
        ["test_a_run_with_no_rewrite_rounds_says_so_rather_than_rendering_nothing"],
        "an empty section and a section saying 'no rounds' are different documents; 4 of 19 runs used "
        "0-2 of their rewrite rounds, so this is the common case",
    ),
    (
        "the Gables ranking follows the sequence order instead of the throughput",
        CR,
        "    return [name for name, _ in sorted(steps, key=lambda kv: -kv[1])]",
        "    return [name for name, _ in steps]",
        ["test_the_gables_four_step_ordering_is_reproduced",
         "test_the_ordering_rule_ranks_by_achieved_throughput_not_by_resource_movement"],
        "three of the four Gables steps are locally sensible moves that made things WORSE, so any rule "
        "that follows the sequence (or one resource) gets the ranking wrong -- which is the point of "
        "using it as a control",
    ),
    (
        "the Gables example replaced by a monotone one",
        CR,
        '    ("step-1", 1.3),\n    ("step-2", 2.0),\n    ("step-3", 160.0),',
        '    ("step-1", 50.0),\n    ("step-2", 80.0),\n    ("step-3", 160.0),',
        ["test_the_gables_sequence_is_non_monotone_which_is_what_makes_it_a_control",
         "test_the_gables_four_step_ordering_is_reproduced"],
        "a monotone example is passable by ANY rule, so the control stops controlling anything while "
        "still reading green",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (CR, REP)}
    ok = True
    unstable: list[str] = []
    try:
        _, _, green, out = run([])
        if not green:
            print("BASELINE IS NOT GREEN -- nothing below means anything\n" + out[-3000:])
            return 2
        print("baseline: green (%s)\n" % out.strip().splitlines()[-1])

        for label, path, old, new, must_fail, why in VARIANTS:
            text = originals[path]
            if text.count(old) != 1:
                print("**SKIPPED** %s: anchor occurs %d times, not once -- the variant would not be "
                      "the change it claims to be" % (label, text.count(old)))
                ok = False
                continue

            # A variant that does not parse fails every test for the WRONG reason, which would read as
            # discrimination. Caught before the run rather than trusted: a syntax error in a patch is
            # the easiest way for this harness to produce a false `ok`.
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
        print("\nVERDICT: harness UNSTABLE on %d variant(s): %s"
              % (len(unstable), ", ".join(unstable)))
        return 1
    if not ok:
        print("\nVERDICT: at least one test is not evidence -- see **FAIL** above")
        return 1
    print("\nVERDICT: every test fails on the implementation it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
