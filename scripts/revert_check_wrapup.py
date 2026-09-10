"""Revert-check for the wrap-up checker: every test must FAIL on the broken reader.

The readers here have one failure mode and it is the dangerous kind: a wrong key path returns a
clean zero rather than an error. At wrap-up that is worse than no checker, because "0 rounds carry
`conversion`" is the exact string `docs/preflight-control-run.md` says to treat as a NEW DEFECT --
so a broken reader manufactures a defect report, and a right one that reads `tuned_ms` instead of
`final_reeval_ms` can invert the J2-5 verdict by more than the noise floor it compares against.

Two of these variants are not hypothetical: they are the two bugs the first draft of
`check_wrapup.py` actually had, found by running it against a finished corpus run.

Same contract and hazards as `revert_check_s2.py` (see its `run()` docstring for why `-B` and `-v`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "check_wrapup.py"
TESTS = [ROOT / "tests" / "test_check_wrapup.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "THE REAL BUG #1: look for a FINAL_REEVAL_DONE event that does not exist",
        CHK,
        '        if t == "RUN_FINISHED":\n'
        "            b = ((p.get(\"summary\") or {}).get(\"best\")) or {}\n"
        "            if b:\n"
        "                best = b",
        '        if t == "FINAL_REEVAL_DONE":\n'
        "            b = p.get(\"best\") or {}\n"
        "            if b:\n"
        "                best = b",
        ["test_the_final_reeval_is_found_in_run_finished",
         "test_the_checker_runs_against_a_real_corpus_run_if_present"],
        "the first draft of this script, verbatim. There is no `FINAL_REEVAL_DONE` event -- "
        "`final_reeval_ms` is written into `RUN_FINISHED.payload.summary.best` by `_finalize`. The "
        "reader returned None for a FINISHED corpus run that has the number, so at wrap-up it "
        "would have reported 'the run produced no result' for a completed 12 h experiment",
    ),
    (
        "THE REAL BUG #2: read BASELINE_DONE without its nesting",
        CHK,
        "            bl = p.get(\"baseline\") or p\n"
        "            lat = bl.get(\"latency_ms\") or {}\n"
        "            name = bl.get(\"kind\") or \"?\"",
        "            lat = p.get(\"latency_ms\") or {}\n"
        "            name = p.get(\"kind\") or \"?\"\n"
        "            bl = p",
        ["test_the_baseline_is_read_through_its_nesting",
         "test_the_checker_runs_against_a_real_corpus_run_if_present"],
        "the second bug the first draft had. `BASELINE_DONE` nests under `payload.baseline`, the "
        "same one-level nesting as TRIAL_DONE. Reading `payload[\"latency_ms\"]` prints '-' for the "
        "baselines, which looks like a run that never measured one -- and the baselines are the "
        "denominator of every speedup in the report",
    ),
    (
        "tuned_ms silently substituted when the re-eval is missing",
        CHK,
        "    reeval = best.get(\"final_reeval_ms\")",
        "    reeval = best.get(\"final_reeval_ms\") or best.get(\"tuned_ms\") or trial_best",
        ["test_tuned_ms_is_never_substituted_for_the_reeval"],
        "the substitution J2-5 exists to forbid. `tuned_ms` is optimistic by 1.5-6.7% while the "
        "noise floor being compared against is 2.35%, so a fallback can flip the arm comparison's "
        "sign. It is also the friendliest-looking possible change: the checker stops saying 'NONE "
        "YET' and starts printing a number",
    ),
    (
        "the arm comparison decides anyway when one re-eval is absent",
        CHK,
        "    if c is None or t is None:",
        "    if c is None and t is None:",
        ["test_the_arm_comparison_refuses_to_decide_without_both_reevals"],
        "compares one arm's real number against `None`, which either raises inside the verdict or "
        "-- worse -- compares against a substituted `tuned_ms` from the variant above. A verdict on "
        "half the data is what 24 h of GPU time was spent to avoid",
    ),
    (
        "the noise floor dropped: any regression counts as a failure",
        CHK,
        "    allowed = c * (1.0 + noise_floor_pct / 100.0)",
        "    allowed = c",
        ["test_the_arm_comparison_uses_the_noise_floor_in_both_directions"],
        "turns run-to-run noise into a finding. The measured L3:43 floor is 2.35% and G9's three "
        "arms landed inside 1.84% of each other, so a zero-tolerance comparison would have called "
        "that a difference. The same test asserts the other direction, which is why a floor of "
        "infinity would not pass either",
    ),
    (
        "0-of-0 rounds reported as the new defect",
        CHK,
        "    if rounds == 0:\n"
        "        verdict = \"NOT YET DECIDABLE -- 0 rewrite rounds, so 0-of-0 says nothing\"\n"
        "    elif with_conv == 0:",
        "    if with_conv == 0:",
        ["test_zero_rounds_is_not_reported_as_a_defect"],
        "manufactures a defect for every run that never reached a rewrite round -- including a run "
        "stopped early, which is 4 of the 5 corpus runs. The preflight rule is 'if STILL 0 that is "
        "a NEW defect', and 'still' means after rounds happened. A checker that cries defect on "
        "every incomplete run is a checker whose output stops being read",
    ),
    (
        "a partial conversion rounded up to PASS",
        CHK,
        "    elif with_conv < rounds:",
        "    elif False:",
        ["test_a_partial_conversion_is_not_rounded_up_to_pass"],
        "some rounds carrying the verdict and some not is the G44 shape -- computed, journalled, "
        "read zero times -- and it is the state a half-wired fix leaves behind. Reporting it as "
        "PASS is how G21 came back the same day it was fixed",
    ),
    (
        "a non-derivable floor counted as a below-floor violation",
        CHK,
        "                if b.get(\"below_floor\"):\n"
        "                    below_floor += 1",
        "                if b.get(\"floor\") is None or b.get(\"below_floor\"):\n"
        "                    below_floor += 1",
        ["test_a_non_derivable_floor_is_not_counted_as_below_floor"],
        "`floor: null` is the HONEST answer for occupancy and registers -- no closed form, and the "
        "sign itself is unreliable (13 non-monotone slices measured). Counting those as violations "
        "reports many per run and buries the real ones, which are the well-fused candidates reading "
        "below `compulsory_bytes`",
    ),
    (
        "an empty precision string counted as naming a precision",
        CHK,
        "                if prov.get(\"precision\"):",
        "                if \"precision\" in prov:",
        ["test_a_reading_pinned_exactly_on_the_floor_is_flagged_for_inspection"],
        "every record carries the KEY -- the ones that are hardware limits or definitional carry it "
        "empty. Counting key-presence reports 100% provenance-with-precision on a run where none "
        "has it, which is precisely the check S3 exists to make (the 107.8% incident was an fp16 "
        "kernel scored against a tf32 ceiling)",
    ),
    (
        "the clamp flag never raised",
        CHK,
        "                if b.get(\"floor\") is not None and b.get(\"room\") == 0.0:\n"
        "                    clamped_suspicion += 1",
        "                if False:\n"
        "                    clamped_suspicion += 1",
        ["test_a_reading_pinned_exactly_on_the_floor_is_flagged_for_inspection"],
        "a reading pinned exactly at its floor with zero room is what a clamp looks like from "
        "outside, and clamping instead of reporting is the specific S3 behaviour the preflight doc "
        "asks to verify. The flag cannot prove a clamp, which is why it says 'inspect' rather than "
        "concluding -- but silence here is indistinguishable from a clean result",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (CHK,)}
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
            print("!! Some tests SKIP here (the fetched corpus is absent). A skip is not a failure,")
            print("!! so a variant whose named tests all skip is UNVERIFIED on this box.\n")

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
                print("        these tests PASSED on the broken reader, so they are not")
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
    print("\nVERDICT: every test fails on the reader it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
