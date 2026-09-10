"""Revert-check for S2: every test must FAIL on the wrong implementation it was written against.

A test that passes on both the broken and the fixed version is not evidence. This project has a
measured instance of exactly that -- six green tests over a loop spinning 2.05M times, all six having
re-implemented the loop inside the test body. So each variant below is a plausible wrong
implementation (in most cases, the one that was actually written first), and the script asserts that
the named test fails on it and that it fails for the RIGHT reason.

Restores every file in a finally block, and verifies the restore by re-running the suite clean at the
end. Run from the repo root:

    PYTHONPATH=src python scripts/revert_check_s2.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIM = ROOT / "src" / "kernel_optimizer" / "evaluation" / "dimensions.py"
DIG = ROOT / "src" / "kernel_optimizer" / "evaluation" / "digest.py"
CHK = ROOT / "src" / "kernel_optimizer" / "evaluation" / "reading_checks.py"
MOD = ROOT / "src" / "kernel_optimizer" / "agents" / "modules.py"
ORC = ROOT / "src" / "kernel_optimizer" / "control" / "orchestrator.py"
TESTS = [ROOT / "tests" / "test_s2_dimensions.py", ROOT / "tests" / "test_s2_wiring.py"]

# (label, file, old, new, tests that MUST fail, what the wrong version does)
VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "no polarity: one band table for every dimension",
        DIM,
        """    if higher_is_better:
        # Inverted: binding when the value is LOW. Uses its own thresholds because "90% of the
        # occupancy ceiling" would read as binding when it is the opposite.
        if frac < OCC_BINDING_BELOW:
            return "binding"
        if frac < OCC_NEAR_BINDING_BELOW:
            return "near-binding"
        return "slack"
""",
        "",
        ["test_j2_6_low_occupancy_is_binding_while_registers_are_not_at_the_hardware_cap",
         "test_j2_6_two_dimensions_binding_at_once_give_two_records",
         "test_j2_7_a_rewrite_with_different_resource_use_gets_different_records"],
        "occupancy 0.167 bands as slack (0.167 < 0.70), so the low-occupancy wall disappears -- the "
        "measured failure that reported ZERO multi-binding while the counter-example was in the input",
    ),
    (
        "one verdict for every dimension (the label path, restated)",
        DIM,
        '    if frac is None:\n        return "unknown"',
        '    return "slack"\n    if frac is None:\n        return "unknown"',
        ["test_j2_1_two_candidates_with_different_resource_states_get_different_verdict_combinations",
         "test_j2_6_low_occupancy_is_binding_while_registers_are_not_at_the_hardware_cap",
         "test_j2_6_two_dimensions_binding_at_once_give_two_records"],
        "every dimension bands identically, so two candidates with different walls get the same "
        "verdict combination -- v2's 19-of-20-same-label defect reproduced inside the new shape",
    ),
    (
        "digest as a compressor: keep only what is actionable",
        DIG,
        "    for rec in state.records:\n        if rec.measured is None and rec.applicable:",
        ("    for rec in state.records:\n"
         "        if rec.verdict not in ('binding', 'near-binding'):\n"
         "            continue\n"
         "        if rec.measured is None and rec.applicable:"),
        ["test_j2_9_the_digest_emits_one_finding_per_record",
         "test_j2_9_every_dimension_reaches_the_prompt_text_including_the_slack_ones",
         "test_a_measured_dimension_with_no_ceiling_is_not_rendered_as_not_measured",
         "test_a_reduced_confidence_reading_says_why_in_the_finding"],
        "2 of 8 dimensions survive -- the natural reading of the plan's first wording "
        "('compressor'), and v2's original defect restated",
    ),
    (
        "no G28 gate at the prompt boundary",
        DIG,
        "    if isinstance(value, (DimensionRecord, DimensionState)):",
        "    if False and isinstance(value, (DimensionRecord, DimensionState)):",
        ["test_n2_a_dimension_record_into_the_prompt_path_raises"],
        "a raw per-dimension record reaches the prompt path silently",
    ),
    (
        "no G28 gate on the raw evidence dict",
        DIG,
        "        hit = raw_markers & set(value)\n        if hit:",
        "        hit = raw_markers & set(value)\n        if False and hit:",
        ["test_n2_raw_evidence_dict_into_the_prompt_path_raises"],
        "the evidence dict -- 31 keys including two that are 1/latency in disguise -- reaches the "
        "prompt",
    ),
    (
        "no applicable-consistency check (the pre-S2 state)",
        CHK,
        "    notes: list[str] = []\n    for rec in records:\n        dim = getattr(rec,",
        "    return []\n    notes: list[str] = []\n    for rec in records:\n        dim = getattr(rec,",
        ["test_n1_applicable_false_with_a_measured_value_is_reported",
         "test_n1_applicable_false_without_a_reason_is_reported",
         "test_applicable_true_with_a_ceiling_and_no_band_is_still_reported",
         "test_a_verdict_with_no_measurement_behind_it_is_reported",
         "test_a_not_applicable_verdict_without_a_reason_is_reported"],
        "grep-confirmed 0 occurrences of `applicable` in reading_checks.py, i.e. the state the plan "
        "recorded as an open pre-condition",
    ),
    (
        "rule 2 alone removed: the flag's own message",
        CHK,
        '            if not reason:\n                notes.append(\n                    "%s is marked applicable=False with no reason,',
        '            if False and not reason:\n                notes.append(\n                    "%s is marked applicable=False with no reason,',
        ["test_n1_applicable_false_without_a_reason_is_reported"],
        "rule 5 still fires on the same input, so a test asserting only 'some note appeared' would "
        "pass -- this variant is what forced that test onto rule 2's own wording",
    ),
    (
        "no reason required on a not-applicable verdict",
        CHK,
        '        if verdict == "not-applicable" and not reason:',
        '        if False and verdict == "not-applicable" and not reason:',
        ["test_a_not_applicable_verdict_without_a_reason_is_reported"],
        "`not-applicable` has two causes (absent dimension / no polarity) and neither is "
        "distinguishable",
    ),
    (
        "ceiling reachability assumed rather than derived",
        DIM,
        "    reach = evidence.get(\"backend_reachable_frac\")\n"
        "    if isinstance(reach, (int, float)) and reach < 1.0:",
        "    reach = evidence.get(\"backend_reachable_frac\")\n"
        "    if False and isinstance(reach, (int, float)) and reach < 1.0:",
        ["test_j2_4_a_backend_unreachable_roof_is_named_with_its_measured_fraction",
         "test_j2_4_an_unreachable_roof_never_appears_in_the_ranked_recommendation"],
        "84.1% of an unreachable roof is presented as ordinary headroom -- the L3:48 accident, where "
        "8/8 tensor-core candidates were rejected while the framework kept aiming at that roof",
    ),
    (
        "measured-but-unbandable rendered as NOT MEASURED",
        DIG,
        "            if f.measured_present:",
        "            if False and f.measured_present:",
        ["test_a_measured_dimension_with_no_ceiling_is_not_rendered_as_not_measured"],
        "the prompt states something false about the box: aten traffic WAS measured, and 'go measure "
        "it' is a different action from 'there is nothing to compare it to'",
    ),
    (
        "a missing evidence key read as zero",
        DIM,
        "    regs = evidence.get(\"n_regs\")\n    max_regs = dev.get(\"max_regs_per_thread\")",
        "    regs = evidence.get(\"n_regs\") or 0\n    max_regs = dev.get(\"max_regs_per_thread\")",
        ["test_a_dimension_whose_evidence_key_is_missing_is_never_read_as_zero"],
        "0 registers bands as slack and reports register headroom on a candidate nobody measured -- "
        "the field-path failure this project has hit repeatedly",
    ),
    (
        "threads_launched marked inapplicable instead of unbandable",
        DIM,
        "        verdict=\"unknown\" if threads is None else \"not-applicable\",\n"
        "        applicable=True,",
        "        verdict=\"unknown\" if threads is None else \"not-applicable\",\n"
        "        applicable=threads is None,",
        ["test_threads_launched_is_never_banded_because_it_has_no_polarity"],
        "says the box does not have the dimension, which is false and hides a real reading behind a "
        "flag that means 'absent'",
    ),
    # --- G28: the defect is in modules.py, so a variant has to reach it -------------------------
    (
        "G28 not closed: the digest is built but the label section still renders",
        MOD,
        "    if digest_text:\n        # S2 vector mode.",
        "    if False and digest_text:\n        # S2 vector mode.",
        ["test_vector_mode_removes_the_label_and_every_raw_evidence_key",
         "test_the_analyst_sandbox_document_is_the_digested_one_in_vector_mode"],
        "the state that G28 describes: `verdict.evidence` rendered key by key into the analyst's "
        "document plus `## Verdict: **{kind}**` -- the exact opposite of J2-3, and invisible to any "
        "test that only drives `for_prompt`. NOTE `test_vector_mode_keeps_the_unmeasurable_list` is "
        "deliberately NOT named here: the old label path renders that list too, so it correctly "
        "survives this variant. It is evidence for the next variant, not this one",
    ),
    (
        "vector mode drops the unmeasurable list along with the label",
        MOD,
        "        if verdict is not None and verdict.unmeasured:",
        "        if False and verdict is not None and verdict.unmeasured:",
        ["test_vector_mode_keeps_the_unmeasurable_list"],
        "removes a caveat rather than changing a form, so the control run's two arms would differ in "
        "two ways at once and J2-5 could not attribute a difference",
    ),
    (
        "the vector is journalled only in vector mode (control arm left with nothing)",
        ORC,
        "            if self.cfg.v3.diagnosis.mode != \"vector\":\n                return None\n"
        "            return for_prompt(d)",
        "            if self.cfg.v3.diagnosis.mode != \"vector\":\n"
        "                self.store.events.pop()\n"
        "                return None\n"
        "            return for_prompt(d)",
        ["test_the_vector_is_journalled_in_label_mode_but_does_not_reach_the_prompt"],
        "recording is made conditional on the prompt arm, so the control run has no vector to "
        "compare against and J2-1 would need a second run to be checkable at all",
    ),
    (
        "a diagnostic failure is allowed to kill the candidate's analysis",
        ORC,
        "        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never fail a run\n"
        "            self.store.append(\"DIMENSION_STATE_FAILED\", {",
        "        except ValueError as exc:  # noqa: BLE001 -- a diagnostic must never fail a run\n"
        "            self.store.append(\"DIMENSION_STATE_FAILED\", {",
        ["test_a_broken_evidence_dict_journals_a_failure_and_does_not_raise"],
        "a defect in the DIAGNOSTIC propagates out and ends the analysis step, so it presents as a "
        "candidate defect -- the inversion several of these fixes exist to prevent, and how "
        "run-l1-42 died at its first analyst call",
    ),
]


def run(names: list[str]) -> tuple[set[str], set[str], bool, str]:
    """(failed, skipped, all-green, raw output) for `names`.

    Uses `-v`, whose per-test lines carry the node id AND the outcome word, plus the RETURN CODE for
    green/not-green. Three earlier versions each got this wrong by reading a status off text that does
    not carry it: `-q -q` suppresses the summary line entirely (a green suite read as "not green"), a
    substring guess could not tell a skip from a pass, and `-rs`'s skip lines are `file:line`, not node
    ids -- so a skipped test looked like one that passed on the broken code. Reporting a skip as
    "passed on the wrong implementation" is a false accusation about a test; reporting it as ok is a
    false clean bill. The outcome has to come from a line that names the test.

    `PYTHONDONTWRITEBYTECODE` is REQUIRED, not hygiene. Python validates a `.pyc` against the source's
    (mtime, size), and this script rewrites one file per variant in quick succession. Several variants
    insert exactly `"False and "` -- the same 10 characters -- so their patched files have IDENTICAL
    SIZE, and when two writes land in the same mtime tick the interpreter reuses the FIRST variant's
    bytecode for the SECOND variant's source. Observed live on the A800: the same variant reported
    `ok` on one invocation and `**FAIL**` on the next, i.e. the harness gave a confident wrong verdict
    about whether a test is evidence. Same family as a broken probe returning a credible constant --
    nothing errors, the number is just wrong.
    """
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
    """Patch, run, restore. Returns (failed, skipped, output).

    Also stales the `.pyc` deliberately by bumping mtime, belt-and-braces alongside `-B`: the failure
    this guards against is bytecode reuse across two same-sized sources written in one mtime tick.
    """
    path.write_text(text.replace(old, new), encoding="utf-8")
    try:
        failed, skipped, _, out = run(names)
    finally:
        path.write_text(text, encoding="utf-8")
    return failed, skipped, out


def main() -> int:
    originals = {p: p.read_text(encoding="utf-8") for p in (DIM, DIG, CHK, MOD, ORC)}
    ok = True
    unverified = 0
    unstable: list[str] = []
    try:
        _, _, green, out = run([])
        if not green:
            print("BASELINE IS NOT GREEN -- nothing below means anything\n" + out[-3000:])
            return 2
        summary = out.strip().splitlines()[-1]
        print("baseline: green (%s)\n" % summary)
        if " skipped" in summary or "SKIPPED" in out:
            print("!! Some tests SKIP on this box (no optuna/torch). A skip is not a failure, so a")
            print("!! variant whose named tests are skipped is UNVERIFIED here -- re-run on the")
            print("!! A800, which is the authoritative suite. Flagged per variant below.\n")

        for label, path, old, new, must_fail, why in VARIANTS:
            text = originals[path]
            if text.count(old) != 1:
                print("**SKIPPED** %s: anchor occurs %d times, not once -- the variant would not be "
                      "the change it claims to be" % (label, text.count(old)))
                ok = False
                continue

            # A variant that does not parse fails every test for the WRONG reason, which would read as
            # discrimination. Caught before the run rather than trusted: a syntax error in a patch is
            # the easiest way for this harness to produce a false `ok`. Added after `revert_check_s4`
            # caught exactly that in one of its own variants.
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
                # Repeat before accusing. A verdict that flips between invocations is a harness
                # defect, not a defective test, and the two need opposite responses -- this script
                # HAS produced a confident wrong verdict that way (stale .pyc, see `run`), so a
                # disagreement between two identical runs must be reported as instability rather
                # than as either answer.
                failed2, _, _ = apply_variant(path, text, old, new, must_fail)
                still = [n for n in wrongly_passed if n not in failed2]
                if len(still) != len(wrongly_passed):
                    unstable.append(label)
                    ok = False
                    print("**UNSTABLE** %s" % label)
                    print("        two identical runs of this variant DISAGREED, so neither answer")
                    print("        can be reported: %s" % ", ".join(
                        n for n in wrongly_passed if n not in still))
                    print("        Fix the harness before reading any verdict here.")
                    print("        wrong version: %s" % why)
                    print()
                    continue
                ok = False
                print("**FAIL** %s" % label)
                print("        these tests PASSED on the wrong implementation, so they are not")
                print("        evidence for it: %s" % ", ".join(wrongly_passed))
            elif unrun and len(unrun) == len(must_fail):
                unverified += 1
                print("UNVERIF %s" % label)
                print("        every named test SKIPPED here (not passed): %s" % ", ".join(unrun))
                print("        verify on the A800.")
            else:
                verified = [n for n in must_fail if n in failed]
                print("ok      %s" % label)
                print("        %d/%d named tests failed as required%s" % (
                    len(verified), len(must_fail),
                    "" if not unrun else " (%d skipped here, verify on the A800)" % len(unrun)))
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
        print("\nVERDICT: the harness is UNSTABLE on %d variant(s) -- no verdict is trustworthy "
              "until that is fixed: %s" % (len(unstable), ", ".join(unstable)))
        return 1
    if not ok:
        print("\nVERDICT: at least one test is not evidence -- see **FAIL** above")
        return 1
    if unverified:
        print("\nVERDICT: every variant this box could check discriminates; %d UNVERIFIED here "
              "and must be re-run on the A800" % unverified)
        return 0
    print("\nVERDICT: every test fails on the implementation it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
