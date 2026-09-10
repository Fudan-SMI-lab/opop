"""Revert-check for S3: every test must FAIL on the wrong implementation it names.

Same contract and hazards as `revert_check_s2.py` -- see that file's `run()` docstring for why `-B` is
required and why outcomes are read from a line that names the test.

The variants here are weighted towards ONE failure family, because it is the one S3 exists to prevent:
inventing a floor. Four of them fabricate a bound in a way that looks entirely reasonable, and each
would make a dimension read as exhausted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BND = ROOT / "src" / "kernel_optimizer" / "evaluation" / "bounds.py"
DIM = ROOT / "src" / "kernel_optimizer" / "evaluation" / "dimensions.py"
DIG = ROOT / "src" / "kernel_optimizer" / "evaluation" / "digest.py"
TESTS = [ROOT / "tests" / "test_s3_bounds.py", ROOT / "tests" / "test_s2_dimensions.py",
         ROOT / "tests" / "test_s2_wiring.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "a register floor invented as zero",
        BND,
        '    if dimension_id in _NOT_DERIVABLE:\n        return LowerBound(source="none", reason=_NOT_DERIVABLE[dimension_id])',
        '    if dimension_id in _NOT_DERIVABLE:\n'
        '        _r, _f, _b = _room(measured, 0.0)\n'
        '        return LowerBound(floor=0.0, source="definitional",\n'
        '                          reason=_NOT_DERIVABLE[dimension_id],\n'
        '                          room=_r, room_frac=_f, below_floor=_b)',
        ["test_registers_report_unknown_and_say_the_sign_is_unreliable",
         "test_shared_memory_reports_unknown_and_cites_the_refuted_formula",
         "test_occupancy_reports_unknown_because_it_is_an_output_not_an_input",
         "test_threads_launched_reports_unknown_for_lack_of_polarity_not_lack_of_data",
         "test_unknown_is_the_majority_answer_and_that_is_the_point",
         "test_the_room_left_figures_reach_the_prompt_with_their_unknowns_intact"],
        "a floor of 0 is the most tempting default and the strongest possible claim -- it says the "
        "dimension can be eliminated entirely. Registers cannot be: the SIGN of their response to a "
        "knob is unreliable (13 non-monotone slices)",
    ),
    (
        "an unmeasured task cost read as a floor of zero",
        BND,
        '        if not compulsory:\n            return LowerBound(source="none", reason=(\n'
        '                "the task\'s compulsory traffic was not measured on this run, so there is no floor to "\n'
        '                "compare against -- not that the floor is zero"))',
        '        if not compulsory:\n            compulsory = 0',
        ["test_a_floor_is_absent_rather_than_zero_when_the_task_cost_was_not_measured"],
        "\"you can eliminate all traffic\" -- the strongest possible claim, made from no data at all",
    ),
    (
        "a reading below its floor clamped to zero room",
        BND,
        "    room = measured - floor",
        "    room = max(0.0, measured - floor)",
        ["test_a_reading_below_its_floor_is_flagged_and_not_clamped"],
        "says 'you are at the floor, 0% room left' about a candidate that is simply not measurable "
        "this way -- aten traffic is itself a lower bound, so a well-fused candidate reads below "
        "compulsory_bytes. A clamp preserves the wrong action",
    ),
    (
        "the weak allocation floor presented at full confidence",
        BND,
        "            room=room, room_frac=frac, below_floor=below,\n"
        "            # Reduced on purpose: a bound on a neighbouring quantity is not a bound on this one, and\n"
        "            # a bare 1.0 here would invite reading the room figure as exact.\n"
        "            confidence=0.5)",
        "            room=room, room_frac=frac, below_floor=below)",
        ["test_the_allocation_floor_is_weak_and_says_so_with_reduced_confidence"],
        "a bound on a NEIGHBOURING quantity (compulsory traffic) is presented as a bound on peak "
        "allocation, with the caching allocator sitting between the two",
    ),
    (
        "provenance as a free-form string (what S2 shipped)",
        BND,
        "    def precision_mismatch(self, candidate_precision: str | None) -> str | None:",
        "    def _disabled_precision_mismatch(self, candidate_precision: str | None) -> str | None:",
        ["test_a_precision_mismatch_is_detected_and_names_the_inversion",
         "test_a_matching_precision_produces_no_warning",
         "test_an_unknown_precision_on_either_side_produces_no_warning",
         "test_the_denominator_notes_reach_the_prompt_as_their_own_section"],
        "a sentence saying 'tensor-core ceiling' cannot be ASKED 'of which precision', so the 107.8% "
        "inversion stays invisible -- an fp16 kernel with 40-46% headroom reads as saturated",
    ),
    (
        "a mismatch claimed when one side's precision is unknown",
        BND,
        "        if not candidate_precision or not self.precision:\n            return None",
        "        if not candidate_precision and not self.precision:\n            return None",
        ["test_an_unknown_precision_on_either_side_produces_no_warning"],
        "a false alarm on every candidate whose precision was not detected, which trains the reader "
        "to ignore the real one -- the same error in the opposite direction",
    ),
    (
        "the precision assumed rather than read from the classifier's label",
        DIM,
        '    for name in ("fp16", "bf16", "tf32", "fp32"):\n'
        '        if name in label:\n'
        '            precision = name\n'
        '            break',
        '    precision = "tf32" if label else ""',
        ["test_the_compute_provenance_extracts_the_precision_from_the_classifiers_own_label"],
        "the same class of guess as guessing an events payload path, which has misfired twice here -- "
        "and it would report every fp16 candidate as matching a tf32 roof, i.e. reproduce the "
        "incident exactly",
    ),
    (
        "a device-query ceiling stamped with the calibration date",
        DIM,
        "            measured_at=cal_at if dated else \"\",\n"
        "            calibration_identity=cal_identity if dated else \"\")",
        "            measured_at=cal_at,\n"
        "            calibration_identity=cal_identity)",
        ["test_a_device_query_ceiling_is_not_stamped_with_a_calibration_date"],
        "implies a recalibration could change the register cap. A hardware limit is not invalidated "
        "by recalibrating",
    ),
    (
        "the self-contradicting tensor-core caveat",
        DIM,
        '        if "tensor-core" in str(ceiling_used):',
        '        if False and "tensor-core" in str(ceiling_used):',
        ["test_a_no_tensor_core_kernel_against_a_tensor_core_roof_is_reported_as_inconsistent"],
        "the sentence says the tensor-core roof is NOT the denominator and then names it as the "
        "denominator. A self-contradicting sentence is worse than either half: a reader cannot act "
        "on it at all",
    ),
    (
        "the room-left section never rendered",
        DIG,
        "    if any(f.room_left for f in d.findings):",
        "    if False and any(f.room_left for f in d.findings):",
        ["test_the_room_left_figures_reach_the_prompt_with_their_unknowns_intact"],
        "a bound nothing renders is not implemented -- the same argument that made `conversion`'s "
        "zero consumers a gap",
    ),
    (
        "room-left computed for unmeasured dimensions too",
        DIG,
        "            room_left=(describe(rec.dimension_id, rec.bound)\n"
        "                       if rec.bound is not None and rec.measured is not None else \"\"),",
        "            room_left=(describe(rec.dimension_id, rec.bound)\n"
        "                       if rec.bound is not None else \"\"),",
        ["test_room_left_is_absent_for_an_unmeasured_dimension_rather_than_unknown"],
        "two sentences about one absence, and the weaker ('room unknown') dilutes the stronger "
        "('never measured')",
    ),
    (
        "the bound dropped from the record",
        DIM,
        "    bound = lower_bound(dimension_id, measured, task_cost)",
        "    bound = None",
        ["test_every_record_carries_a_provenance_and_a_bound",
         "test_the_bound_is_journalled_inside_the_record",
         "test_the_room_left_figures_reach_the_prompt_with_their_unknowns_intact",
         "test_the_s3_bound_and_provenance_reach_the_journalled_record"],
        "the floor exists only in a prompt and cannot be re-checked offline -- and every J2/J3 "
        "criterion is verified by replay",
    ),
    (
        "a task cost required rather than optional",
        BND,
        "    compulsory = getattr(task_cost, \"compulsory_bytes\", None) if task_cost is not None else None",
        "    compulsory = task_cost.compulsory_bytes",
        ["test_a_floor_is_absent_rather_than_zero_when_the_task_cost_was_not_measured",
         "test_a_run_without_a_calibration_or_task_cost_still_produces_a_vector"],
        "a box that cannot be calibrated, or a run whose task-cost measurement failed, loses its "
        "whole resource vector -- and both are legitimate states, not errors. NOTE the definitional "
        "floors (spills, aten ops) correctly SURVIVE this variant: they return before `task_cost` is "
        "read, so naming their test here would have been naming a test that should not fail",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (BND, DIM, DIG)}
    ok = True
    unverified = 0
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
        print("\nVERDICT: harness UNSTABLE on %d variant(s): %s"
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
