"""Revert-check for S2d: every test must FAIL on the wrong implementation it names.

Same contract and the same hazards as `revert_check_s2.py` -- see that file's `run()` docstring for
why `-B` is required (same-sized patched sources plus a one-tick mtime collision made this harness
report a confident wrong verdict on the A800) and why the outcome is read from a line that names the
test rather than from a summary line a flag can silence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "src" / "kernel_optimizer" / "evaluation" / "reconcile.py"
REP = ROOT / "src" / "kernel_optimizer" / "models" / "reports.py"
MOD = ROOT / "src" / "kernel_optimizer" / "agents" / "modules.py"
ORC = ROOT / "src" / "kernel_optimizer" / "control" / "orchestrator.py"
TESTS = [ROOT / "tests" / "test_s2d_reconcile.py", ROOT / "tests" / "test_s2d_wiring.py"]

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "measured direction derived from conversion's polarity-aware `direction`",
        REC,
        '    delta = delta_info.get("delta")\n    rel = delta_info.get("rel")',
        '    _d = delta_info.get("direction")\n'
        '    if _d in ("improved", "worsened"):\n'
        '        return "down" if _d == "improved" else "up"\n'
        '    delta = delta_info.get("delta")\n    rel = delta_info.get("rel")',
        ["test_j2d_2_occupancy_going_up_is_up_even_though_up_is_better_for_it"],
        "`up`/`down` become polarity-aware, so occupancy -- the one dimension where higher is better "
        "-- inverts. This is the error family that reported ZERO multi-dimension binding with the "
        "counter-example in the input",
    ),
    (
        "unpredicted folded into misses",
        REC,
        '                match="unpredicted", before=info.get("before"),',
        '                match="miss", before=info.get("before"),',
        ["test_j2d_2_a_dimension_that_moved_and_was_not_mentioned_is_unpredicted_not_a_miss"],
        "'thought wrong' and 'did not think of it' are pooled, so the next round is told the agent "
        "was wrong about something it never claimed",
    ),
    (
        "a missing measurement read as flat",
        REC,
        '        actual = _actual_direction(info) if info else "unknown"',
        '        actual = _actual_direction(info) if info else "flat"',
        ["test_a_declared_dimension_with_no_measurement_is_unmeasured_not_flat"],
        "an agent that says `unchanged` about a dimension nobody looked at is scored a HIT -- free "
        "credit manufactured out of a gap in the measurement",
    ),
    (
        "`unknown` silently skipped",
        REC,
        '        expected: Expect = getattr(exp, "expect", "unknown") or "unknown"',
        '        expected: Expect = getattr(exp, "expect", "unknown") or "unknown"\n'
        '        if expected == "unknown":\n'
        '            continue',
        ["test_j2d_6_an_all_unknown_round_leaves_a_visible_zero_information_record"],
        "an agent can answer `unknown` to everything forever and never accumulate a record of it "
        "(J2d-6's stated failing condition)",
    ),
    (
        "no caveat on the ledger entry",
        REC,
        "        caveat=CAVEAT)",
        '        caveat="")',
        ["test_j2d_7_every_entry_carries_the_retuning_caveat",
         "test_the_caveat_survives_into_the_rendered_ledger",
         "test_the_caveat_is_one_shared_string_not_per_entry_prose"],
        "a `miss` blames the agent for the TUNER's choice -- a rewrite is re-tuned, so the comparison "
        "is between two different configurations. An agent adjusting to wrong feedback is worse off "
        "than one with none (1.77x against 3.35x, p=0.0007)",
    ),
    (
        "the ledger rendered as a JSON dump",
        REC,
        '    if not entries:\n        return ""',
        '    import json as _json\n    return _json.dumps(entries, indent=2)\n'
        '    if not entries:\n        return ""',
        ["test_j2d_5_the_rendering_is_one_line_per_dimension_per_round",
         "test_j2d_5_the_rendering_grows_linearly_in_rounds",
         "test_the_caveat_survives_into_the_rendered_ledger"],
        "the ledger becomes a second raw vector, reproducing KernelPro's losing arm; G9 measured "
        "volume buys nothing at 2.1x the cost",
    ),
    (
        "the vocabulary restated instead of shared",
        REC,
        "DIMENSION_VOCABULARY: tuple[str, ...] = tuple(sorted(_DIMENSIONS))",
        'DIMENSION_VOCABULARY: tuple[str, ...] = ("n_regs", "shared_bytes", "occupancy")',
        ["test_the_vocabulary_is_the_shared_one_not_a_second_list",
         "test_every_vocabulary_name_can_actually_be_reconciled"],
        "a second list drifts: it keeps a dimension after `conversion._DIMENSIONS` drops it, so the "
        "schema accepts a name the reconciler can never compare",
    ),
    (
        "an invented magnitude field silently dropped",
        REP,
        '    model_config = ConfigDict(extra="forbid")\n',
        "",
        ["test_an_invented_magnitude_field_is_rejected_rather_than_silently_dropped"],
        "pydantic's DEFAULT: `expected_pct: 40` validates and the 40 vanishes, so the agent reasoned "
        "from a magnitude nobody ever checked -- exactly what the schema exists to prevent",
    ),
    (
        "check_output does not validate the vocabulary",
        MOD,
        "        named = [e.dimension for c in output.candidates for e in c.expectations]\n"
        "        bad = unknown_dimensions(named)\n        if bad:",
        "        named = [e.dimension for c in output.candidates for e in c.expectations]\n"
        "        bad = unknown_dimensions(named)\n        if False and bad:",
        ["test_j2d_1_check_output_rejects_an_unknown_dimension_and_lists_the_legal_ones"],
        "free-text dimension names are accepted, and the expectation is then about something nothing "
        "will ever measure -- the unfalsifiable state `Hypothesis.expected_effect` was already in",
    ),
    (
        "the ledger is built but never seeded into the sandbox",
        MOD,
        "        if inputs.ledger_entries:",
        "        if False and inputs.ledger_entries:",
        ["test_j2d_4_the_ledger_reaches_the_rewriters_sandbox_as_prose"],
        "the rewriter sees 'H1 was tried and did not help' with no way to learn 'H1 said shared "
        "memory would fall and it rose' -- the exact gap S2d exists to close",
    ),
    (
        "the prompt hard-codes the old history filename",
        MOD,
        '        history = ("`history/prediction_ledger.md` shows what you PREDICTED in earlier rounds and "\n'
        '                   "what was measured" if inputs.ledger_entries\n'
        '                   else "`history/failed_hypotheses.json` lists changes already tried that did NOT "\n'
        '                        "help")',
        '        history = ("`history/failed_hypotheses.json` lists changes already tried that did NOT "\n'
        '                   "help")',
        ["test_the_prompt_points_at_whichever_history_file_exists"],
        "the ledger is written and never read: the prompt names a file that is not there, which "
        "teaches the agent to stop opening files",
    ),
    (
        "declarations kept instead of popped",
        ORC,
        "            declared = self.round_expectations.pop(family_id, [])",
        "            declared = self.round_expectations.get(family_id, [])",
        ["test_the_declarations_are_popped_so_the_next_round_is_not_scored_against_them"],
        "round N+1 is scored against round N's predictions -- a plausible-looking ledger that is "
        "entirely wrong, and every hit/miss attributed to the wrong round",
    ),
    (
        "a reconcile failure ends the rewrite round",
        ORC,
        "        except Exception as exc:  # noqa: BLE001 -- a diagnostic must never end a round",
        "        except ValueError as exc:  # noqa: BLE001 -- a diagnostic must never end a round",
        ["test_a_reconcile_failure_journals_and_does_not_end_the_round"],
        "a bookkeeping defect presents as a candidate defect and costs a rewrite round -- the "
        "scarcest budget in the loop (5 completed L3 runs used 9 rounds in total)",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (REC, REP, MOD, ORC)}
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
            print("!! Some tests SKIP on this box (no optuna/torch). A skip is not a failure, so a")
            print("!! variant whose named tests all skip is UNVERIFIED here -- re-run on the A800.\n")

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
