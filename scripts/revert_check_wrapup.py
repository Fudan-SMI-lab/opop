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
        "THE REAL BUG #3: the S3 precision counted per-record instead of per-diagnosis",
        CHK,
        "            cprov = p.get(\"compute_ceiling_provenance\") or {}",
        "            cprov = ((p.get(\"records\") or [{}])[0].get(\"provenance\")) or {}",
        ["test_the_compute_ceiling_precision_is_read_from_the_top_level_field"],
        "the third bug this script had, found on LIVE box-2 data. `_do_diagnose` writes "
        "`compute_ceiling_provenance` BESIDE `records`, because exactly one dimension has a "
        "precision -- the compute-pressure denominator, the only one that can be the wrong "
        "denominator without anything looking wrong (an fp16 kernel against a tf32 ceiling read "
        "107.8% of peak). The per-record blocks are `definitional`/`device_query` and their "
        "precision is legitimately EMPTY, so counting those reports 0 on a run whose S3 field says "
        "`precision: \"fp16\"` with a full calibration identity. Same shape as `FINAL_REEVAL_DONE`: "
        "a clean zero on data that HAS the number, and S3's whole point is that this denominator's "
        "provenance be checkable",
    ),
    (
        "the two precision counts collapsed into one",
        CHK,
        "                if cprov.get(\"precision\"):\n"
        "                    prov_with_precision += 1",
        "                if cprov.get(\"precision\"):\n"
        "                    prov_with_precision += 1\n"
        "                    record_prov_with_precision += 1",
        ["test_the_compute_ceiling_precision_is_read_from_the_top_level_field"],
        "a zero means OPPOSITE things in the two places: 0 per-record precisions is correct and "
        "expected (hardware limits have none), while 0 compute-ceiling precisions on a run with "
        "diagnoses is the S3 failure. Merging the counts destroys the distinction in whichever "
        "direction the reader happens to look, which is how a checked criterion becomes an unchecked "
        "one without any line saying so",
    ),
    (
        "the honest 'no compute roof' case counted as naming a precision",
        CHK,
        "                if cprov.get(\"precision\"):\n"
        "                    prov_with_precision += 1\n"
        "                if cprov.get(\"calibration_identity\"):",
        "                if cprov.get(\"precision\") or cprov.get(\"source\") != \"measured\":\n"
        "                    prov_with_precision += 1\n"
        "                if cprov.get(\"calibration_identity\"):",
        ["test_a_diagnosis_with_no_compute_ceiling_reports_zero_not_a_crash"],
        "an overhead-bound or cannot-run candidate genuinely has no compute roof in force -- "
        "`source: \"none\"`, precision empty -- and that zero is the HONEST reading, not a reader "
        "failure. Treating it as 'names a precision' reports S3 as fully provenanced on a run where "
        "the denominator was never measured, which is the mirror image of real bug #3: there the "
        "reader under-reported a precision that exists, here it over-reports one that does not. "
        "(An earlier version of this variant patched `if cprov:` to `if True:` and did NOT "
        "discriminate -- the `source: \"none\"` provenance is a non-empty dict, so the denominator "
        "was already being counted. Measured, not assumed.)",
    ),
    (
        "the precision_mismatch read from the wrong nesting level",
        CHK,
        "            if p.get(\"precision_mismatch\"):\n"
        "                mismatches.append(p.get(\"candidate_id\"))",
        "            if (p.get(\"digest\") or {}).get(\"precision_mismatch\"):\n"
        "                mismatches.append(p.get(\"candidate_id\"))",
        ["test_the_precision_mismatch_is_read_from_the_top_level_field"],
        "`precision_mismatch` sits at the top level beside the provenance. Reading it out of the "
        "digest returns nothing, so the 107.8%-incident detector reports 'nothing' for every run -- "
        "including one where it fired. This is the criterion that S3 exists to make checkable, and "
        "silence from it is indistinguishable from a clean result",
    ),
    (
        "THE REAL BUG #4: the loop inferred from the WALL_CLOCK_REACHED payload shape",
        CHK,
        "    if rounds == 0:\n"
        "        tail = (\" NO rewrite round ever ran, so a zero in check 1 means 'the run never "
        "got there' \"",
        "    if any(s[\"site\"] == \"candidate batch\" for s in stops):\n"
        "        tail = (\" NO rewrite round ever ran, so a zero in check 1 means 'the run never "
        "got there' \"",
        ["test_a_budget_stop_with_rounds_does_not_claim_loop_c_never_ran"],
        "the fourth bug this script had, caught on the corpus run within a minute of writing it. "
        "`_pipeline_batch` is called for the SEED batch AND from inside Loop C for rewrite "
        "candidates, so a `skipped`-shaped stop does NOT mean the seed pipeline ran out of time. "
        "`run-l3-43-20260909-015247` fired exactly that stop at 13.51 h AND has 5 "
        "FAMILY_ROUND_RECORDED, the last landing in the same second. Inferring the loop from the "
        "payload shape asserts the opposite of the truth on the first real run it sees",
    ),
    (
        "a budget stop with no rounds still reported as 'not yet decidable'",
        CHK,
        "    if rounds == 0:\n"
        "        tail = (\" NO rewrite round ever ran",
        "    if False:\n"
        "        tail = (\" NO rewrite round ever ran",
        ["test_a_budget_stop_with_no_rounds_overrides_not_yet_decidable"],
        "reports the wrong meaning for a run the clock ENDED before Loop C. It is decided: the "
        "answer is 'never got there', not 'not yet decidable'. This is box 3's live risk case -- its "
        "projected 3.53-9.66 h for three remaining candidates against a 10.23 h budget -- so the "
        "distinction decides how its G27 contribution is reported rather than being hypothetical. "
        "(An earlier version of this variant patched the PRINT path in `report()`, which the test "
        "does not exercise because it calls `check_budget_stop` directly -- measured, then moved to "
        "the branch that actually produces the verdict string.)",
    ),
    (
        "only one of the two emission sites recognised",
        CHK,
        "        site = \"candidate batch\" if \"skipped\" in p else (\n"
        "            \"rewrite round\" if \"round\" in p else \"unknown site\")",
        "        if \"skipped\" not in p:\n"
        "            continue\n"
        "        site = \"candidate batch\"",
        ["test_both_emission_sites_are_recognised"],
        "`WALL_CLOCK_REACHED` is emitted from `_pipeline_batch` (payload: pipelined/skipped) and "
        "from `_rewrite_round` (payload: round/stopped_before_family). Handling one drops the other "
        "SILENTLY, and the rewrite-round stop is the one that says a round count is clock-limited "
        "rather than a convergence result",
    ),
    (
        "the overrun size dropped from the verdict",
        CHK,
        "        over = \" (%.0f%% over)\" % (\n"
        "            100.0 * (first[\"elapsed_hours\"] - first[\"budget_hours\"]) / "
        "first[\"budget_hours\"])",
        "        over = \"\"",
        ["test_the_overrun_percentage_is_reported"],
        "the recorded finding is that the wall clock is ALWAYS the binding budget, so by how much a "
        "run overran it is the number that matters -- this corpus run exceeded its 12 h by 13%, "
        "which means every per-run cost figure derived from the configured budget is an "
        "underestimate. Without the percentage the verdict says a stop happened but not that the "
        "budget failed to hold",
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
        "                if cprov.get(\"precision\"):\n"
        "                    prov_with_precision += 1",
        "                if \"precision\" in cprov:\n"
        "                    prov_with_precision += 1",
        ["test_a_diagnosis_with_no_compute_ceiling_reports_zero_not_a_crash"],
        "every provenance carries the KEY -- the ones that are hardware limits or definitional carry it "
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
    (
        "ledger entries counted without checking n_declared",
        CHK,
        "            n = p.get(\"n_declared\")\n"
        "            if not n:\n"
        "                empty += 1\n"
        "            else:\n"
        "                declared_total += int(n)",
        "            declared_total += int(p.get(\"n_declared\") or 0)",
        ["test_an_all_empty_ledger_is_not_reported_as_working",
         "test_a_partly_empty_ledger_is_not_rounded_up"],
        "the S2d trap. `_record_reconciliation` appends an entry even when the rewriter declared "
        "nothing (`n_declared: 0`), so a ledger built entirely of empty entries produces a "
        "perfectly healthy-looking event stream. Counting events reports PASS on a ledger that "
        "reconciles nothing -- the G44 shape (computed, journalled, read zero times) wearing the "
        "clothes of a working feature",
    ),
    (
        "a missing ledger read as 'the switch is off' rather than a defect",
        CHK,
        "    elif entries == 0:\n"
        "        verdict = (\"**DEFECT** -- %d rounds and 0 ledger entries. It is journalled \"",
        "    elif entries == 0 and False:\n"
        "        verdict = (\"**DEFECT** -- %d rounds and 0 ledger entries. It is journalled \"",
        ["test_zero_ledger_entries_with_rounds_is_a_defect_in_either_arm"],
        "the asymmetry that is easy to rationalise away: 'the control arm has the ledger off, so of "
        "course it has no entries'. Wrong -- `_record_reconciliation` journals UNCONDITIONALLY and "
        "the switch gates only whether the rendered form reaches the rewriter's PROMPT. That "
        "asymmetry is deliberate: recording is what makes the control arm analysable from its own "
        "log, so a control arm with rounds and no entries is a defect in both arms' shared path",
    ),
    (
        "a reconcile failure swallowed (it is silent everywhere else)",
        CHK,
        "    if failed:\n"
        "        verdict += \"  || %d RECONCILE_FAILED (silent by design -- a diagnostic must not end a \" \\\n"
        "                   \"round): %s\" % (len(failed), failed[0])",
        "    if False:\n"
        "        verdict += \"  || %d RECONCILE_FAILED (silent by design -- a diagnostic must not end a \" \\\n"
        "                   \"round): %s\" % (len(failed), failed[0])",
        ["test_a_reconcile_failure_is_surfaced_because_it_is_otherwise_silent"],
        "`_record_reconciliation` catches every exception so a diagnostic cannot end a rewrite "
        "round -- correct design, and it means a totally broken reconciler is invisible apart from "
        "this event. Dropping it from the verdict makes the ledger's failure mode unobservable at "
        "exactly the moment someone is deciding whether S2d worked",
    ),
    (
        "no measured latency floor: only the borrowed correctness figure available",
        CHK,
        "    deltas = []\n"
        "    for d in run_dirs:\n"
        "        fin = final_result(d)",
        "    deltas = []\n"
        "    for d in []:\n"
        "        fin = final_result(d)",
        ["test_the_latency_floor_is_measured_from_the_reeval_not_borrowed",
         "test_the_latency_floor_takes_the_WIDEST_delta_across_arms"],
        "the state before this fix: the only tolerance was 2.35%, which is `1 - 0.9765` where "
        "0.9765 is the reference's own frac_within_tol at two precisions -- a fraction of ELEMENTS "
        "agreeing, used as a LATENCY tolerance. A numerics figure and a timing-jitter figure have "
        "no reason to be equal. Measured on the corpus, the same-kernel re-eval delta is 0.27% and "
        "2.99%, so 2.35% happens to land inside the range while being derived from the wrong "
        "quantity -- the most durable kind of wrong number, because it never looks wrong",
    ),
    (
        "the narrower of the two arm deltas chosen instead of the wider",
        CHK,
        "    return max(deltas), \"measured same-kernel re-eval delta, n=%d, widest %.2f%%\" % (\n"
        "        len(deltas), max(deltas))",
        "    return min(deltas), \"measured same-kernel re-eval delta, n=%d, widest %.2f%%\" % (\n"
        "        len(deltas), min(deltas))",
        ["test_the_latency_floor_takes_the_WIDEST_delta_across_arms"],
        "a latency verdict stricter than the latency measurement. With two arms there are two "
        "same-kernel deltas, and taking the narrower calls a difference smaller than the "
        "measurement a regression -- on the corpus numbers, judging against 0.27% when the same "
        "kernel re-measured 2.99% away on another run",
    ),
    (
        "a missing re-eval reported as a floor of zero",
        CHK,
        "    if not deltas:\n"
        "        return None, \"no arm has re-evaluated its best kernel yet\"",
        "    if not deltas:\n"
        "        return 0.0, \"no arm has re-evaluated its best kernel yet\"",
        ["test_no_reeval_yet_reports_the_absence_rather_than_zero"],
        "a floor of 0.0 makes EVERY difference a regression, down to a single microsecond. The "
        "absence of a measurement is not a measurement of zero -- and this is the friendly-looking "
        "version, because it lets the comparison print a verdict instead of refusing to decide",
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
