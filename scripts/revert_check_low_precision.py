"""Revert-check for the low-precision contract change: each test must FAIL on the old wording.

Every variant here restores prose that was ACTUALLY in the prompts before this change, or the most
plausible weaker wording. That matters more than usual for a text-only change: a test asserting a
substring is trivially satisfied, so the only evidence it is worth anything is that removing the
sentence it guards makes it fail.

Same contract and hazards as `revert_check_s2.py` (see its `run()` docstring for why `-B` and `-v`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "src" / "kernel_optimizer" / "agents" / "modules.py"
CON = ROOT / "src" / "kernel_optimizer" / "agents" / "prompts" / "candidate_contract.md"
TESTS = [ROOT / "tests" / "test_low_precision_contract.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "the repair prompt's ORIGINAL wording: ieee as the remedy",
        MOD,
        '        "Do NOT reach for input_precision=\\"ieee\\" as the fix unless the task genuinely "\n'
        '        "needs full fp32: that abandons the tensor cores altogether and is measurably "\n'
        '        "the wrong trade -- on one task 8 of 8 tensor-core candidates were rejected, "\n'
        '        "7 of 7 scalar candidates accepted, and the resulting best kernel used no "\n'
        '        "tensor cores at all."',
        '        "A small error just over tolerance means the dot needs more precision: use "\n'
        '        "input_precision=\\"ieee\\" for tl.dot on fp32 refs."',
        ["test_the_repair_prompt_does_not_prescribe_abandoning_the_tensor_cores"],
        "the prescription that was there before this change, with the warning REMOVED. The old "
        "sentence reads as sound numerical advice and its outcome is on record: 8 of 8 tensor-core "
        "candidates rejected, 7 of 7 scalar accepted, and a best kernel that used no tensor cores "
        "at all. NOTE the first version of this variant only ADDED the ieee prescription while "
        "leaving the warning in place, and the test correctly still passed -- the two can coexist, "
        "so the variant has to remove what the test actually asserts",
    ),
    (
        "the three causes collapsed into one 'precision issue'",
        MOD,
        '        "  (b) AN ALGORITHM BRANCH THAT ONLY ONE PRECISION ENTERS -- if the source has "',
        '        "  (b) another precision issue -- if the source has "',
        ["test_the_repair_prompt_names_all_three_causes_with_distinct_fixes"],
        "the repair agent cannot tell an algorithm branch from a mantissa shortfall, so it guesses "
        "-- and two of the three guesses make the candidate worse (adding split3 to a kernel "
        "overflowing to Inf changes nothing and costs 3x the MMAs)",
    ),
    (
        "the semantic-mismatch guidance displaced by the precision guidance",
        MOD,
        "BatchNorm/normalization run in the wrong mode. Read `task/eval_semantics.md`: \"\n"
        "        \"if the reference is in TRAIN mode, BatchNorm must use the CURRENT BATCH ",
        "a precision problem somewhere in the reduction. \"\n        \"Otherwise ",
        ["test_the_repair_prompt_names_all_three_causes_with_distinct_fixes"],
        "a LARGE constant offset is a different failure with its own measured root cause "
        "(train-mode BatchNorm using running stats on an untrained model, the recorded cause of "
        "repeated L3:21 failures). Adding precision guidance must not push it out: it is the FIRST "
        "thing to check, and this variant is why the test asserts it survived. NOTE the first "
        "version of this variant edited only the words after 'most often' and left both asserted "
        "strings in the block, so the test correctly still passed",
    ),
    (
        "the rule stated as a preference, without its measurement",
        CON,
        "PREC only inside the dot helper      fp16 32/32, bf16 17/17, tf32 12/12  PASS\n"
        "PREC only inside the dot helper      fp16 15/15, bf16 41/41 pass; tf32   0/11 FAIL\n"
        "PREC in the dot helper AND the ALGORITHM   fp16 43/43 pass; bf16 0/13, tf32 0/9  FAIL\n"
        "PREC in the dot helper AND the ALGORITHM   fp16 pass;       bf16 0/12, tf32 0/4  FAIL",
        "PREC confined to the dot helper passes at every precision; PREC that also gates the\n"
        "algorithm fails at whichever precisions you did not debug.",
        ["test_the_contract_forbids_the_precision_knob_from_selecting_a_code_path"],
        "'write the algorithm once' with no numbers reads as tidiness advice, and an agent weighing "
        "it against its own judgement has nothing to weigh. The 0/13 and 0/9 are what make it a "
        "correctness rule. NOTE the first version blanked only the table's FIRST row, while 0/13 "
        "and 0/9 live in the third -- the variant has to remove the whole table",
    ),
    (
        "the remedy omitted: what to do with a precision that needs stabilization",
        CON,
        "**unconditionally at every precision**",
        "**in the branch for that precision**",
        ["test_the_contract_forbids_the_precision_knob_from_selecting_a_code_path"],
        "states the prohibition and then prescribes the forbidden thing. An agent told not to "
        "branch, with no alternative, reasonably concludes the stabilization must go in a branch -- "
        "which is the defect verbatim",
    ),
    (
        "the `if PREC ==` diagnostic dropped",
        CON,
        "by counting `if PREC ==` sites per candidate:",
        "by inspecting each candidate's source:",
        ["test_the_contract_gives_the_diagnostic_that_separates_the_causes"],
        "without a COUNTABLE marker the two causes cannot be separated by reading the source, and "
        "the four-candidate table that established the rule could not have been built. NOTE the "
        "contract's only occurrence of the marker is this table caption, which is what the first "
        "version of this variant got wrong (it named a phrase that is not in the file)",
    ),
    (
        "the compensated dot recommended instead of offered as a knob",
        CON,
        '    "DOT_MODE": "plain",         # ["plain", "split3"]  -- arithmetic only, NOT a code path',
        '    # always use the 3-MMA split form shown above',
        ["test_the_contract_offers_the_compensated_dot_as_a_tunable_knob"],
        "trades one hard-coded choice for another: split3 costs 3x the MMAs, so on a short "
        "reduction or narrow dynamic range it is pure loss. Which form wins is a measurement "
        "question, and hard-wiring either answer removes it from the tuner",
    ),
    (
        "a cross term dropped from the split (the plausible typo)",
        CON,
        "acc += tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)   # 3 MMAs, ~2x mantissa bits",
        "acc += tl.dot(ah, bh) + tl.dot(al, bl)   # 3 MMAs, ~2x mantissa bits",
        ["test_the_split_form_in_the_contract_is_arithmetically_correct"],
        "the snippet is COPIED by agents, so an error propagates into candidates. Keeping (al,bl) "
        "while dropping the two significant cross terms is the plausible version of this mistake -- "
        "it looks symmetric and is nearly worthless, since al*bl is below the result's "
        "representable range while ah*bl and al*bh carry the recovered mantissa bits",
    ),
    (
        "fp16 range folded in with mantissa",
        CON,
        "**A third failure mode is fp16-specific and is about RANGE, not mantissa**",
        "**A third failure mode is another fp16 precision problem**",
        ["test_the_contract_separates_fp16_range_from_mantissa"],
        "an agent adds split3 to a kernel that is overflowing to Inf, measures no improvement, and "
        "has no way to reach the actual fix (rescale, or use bf16, whose exponent range is fp32's)",
    ),
    (
        "the parameterizer told to preserve the structure it is given",
        MOD,
        "that is a defect to FIX while\n   parameterizing, not a structure to preserve",
        "parameterize it faithfully like any other knob",
        ["test_the_parameterizer_is_told_to_fix_a_gating_knob_not_preserve_it"],
        "the parameterizer's contract everywhere else is to parameterize FAITHFULLY, so preserving "
        "an algorithm-gating PREC is the default reading of its job. It is the one module that can "
        "turn this defect into a tuned dimension, and the instruction to hoist has to be explicit",
    ),
    (
        "the parameterizer left without the compensated-dot knob",
        MOD,
        '   expose that as its OWN knob rather than a branch — `"DOT_MODE"` with choices\n'
        '   `["plain", "split3"]`, where `"split3"` splits each operand into high and low\n'
        '   parts and accumulates the cross terms (3 MMAs, ~2x the mantissa bits):',
        '   handle that however seems best:',
        ["test_the_parameterizer_prompt_carries_the_rule[DOT_MODE-the compensated-dot knob]"],
        "'however seems best' is the state before this change: with no named alternative the "
        "natural expression of 'this precision needs more mantissa' is a branch on precision. "
        "NOTE two naming errors of mine here, both worth keeping visible: the first version "
        "softened only the sentence and left `DOT_MODE` in the snippet below it (so the test "
        "correctly still passed), and the second named the BARE test name while this test is "
        "parametrized -- pytest's node id carries the parameter suffix, so the harness matched no "
        "line at all. Only the DOT_MODE case may be named: the COMPUTE_DTYPE and ARITHMETIC-ONLY "
        "cases correctly survive a variant that removes only the compensated-dot knob",
    ),
    (
        "the generator's superseded knob name restored",
        MOD,
        'precision as a PARAMS knob (e.g. "COMPUTE_DTYPE" with choices\n'
        '["fp16", "bf16", "tf32", "ieee"]) so the tuner can compare them on real measurements.',
        'precision as a PARAMS knob (e.g. "DOT_PRECISION": "tf32") so the tuner can compare\n'
        'it against "ieee" on real measurements.',
        ["test_the_generator_prompt_no_longer_names_the_superseded_knob"],
        "the wording that was there before. Two names for one concept across two prompts is how a "
        "candidate ends up with a dtype cast in the body and a mismatched knob in PARAMS -- an "
        "already-measured failure. It also contrasted only tf32 against ieee, leaving bf16 "
        "unmentioned, and bf16 passed on a task where fp16 overflowed",
    ),
    (
        "split3 allowed to fall through to ieee at tf32 (the observed gap)",
        CON,
        "**`DOT_MODE` must be meaningful for every `COMPUTE_DTYPE` that reaches the tensor cores**",
        "**`DOT_MODE` applies where you judge it useful**",
        ["test_the_contract_forbids_split3_falling_through_to_ieee_for_tf32"],
        "the state the prompt was in on the first real run after the contract change: 4 of 4 "
        "candidates implemented DOT_MODE correctly in every other respect and ALL FOUR dropped the "
        "(split3, tf32) pair. 4/4 is the prompt being silent, not chance -- and the tuner then "
        "reports on a combination it never measured. 'Where you judge it useful' is exactly the "
        "licence each agent took",
    ),
    (
        "a task named in the rule (case-specific special-casing)",
        CON,
        "Measured over four candidates on one task, by counting `if PREC ==` sites per candidate:",
        "Measured on L3:48 (Mamba), by counting `if PREC ==` sites per candidate:",
        ["test_no_task_specific_special_casing_entered_the_prompts"],
        "the standard this project holds fixes to: a rule naming one task is guidance about that "
        "task, and S1b was withdrawn for exactly this -- a correlation that held only in the "
        "corpus at hand. The measurement belongs in the docs; the RULE must be general",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (MOD, CON)}
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

            # A .py variant that does not parse fails every test for the WRONG reason, which reads
            # as discrimination (G43). Markdown has no such check available, which is one more
            # reason every variant here must name the tests it expects to break.
            patched = text.replace(old, new)
            if path.suffix == ".py":
                try:
                    compile(patched, str(path), "exec")
                except SyntaxError as exc:
                    print("**SKIPPED** %s: the patched file does not parse (%s), so any failure it "
                          "produced would be for the wrong reason" % (label, exc))
                    ok = False
                    continue

            failed, was_skipped, out = apply_variant(path, text, old, new, must_fail)
            wrongly_passed = [n for n in must_fail if n not in failed and n not in was_skipped]

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
                print("        these tests PASSED on the wrong wording, so they are not")
                print("        evidence for it: %s" % ", ".join(wrongly_passed))
            else:
                verified = [n for n in must_fail if n in failed]
                print("ok      %s" % label)
                print("        %d/%d named tests failed as required" % (len(verified), len(must_fail)))
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
    print("\nVERDICT: every test fails on the wording it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
