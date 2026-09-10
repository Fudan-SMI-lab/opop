"""Revert-check for the witness-fallback ordering: every test must FAIL on the old walk.

This one matters more than usual because the defect it guards was INVISIBLE to a green suite. The
fallback existed, its out-of-range predicate fired correctly on the real failure, and it still
rejected the candidate -- because `itertools.product` varies its LAST factor fastest, so both
retries held the first knob at `choices[0]`, which on the failure this exists for is the dtype that
just overflowed. Nothing about that is wrong at the level of "does the function return"; it is
wrong only in WHICH configs it chooses. So the variants below restore the old choosing.

Two directions, as always:

  TOO WEAK -- the retries stay near the failure and the fallback rejects candidates whose algorithm
  is already repaired. Measured: 9 of 9 candidates declaring a precision knob rejected on one task
  while 7 of 7 without one published.

  TOO BROAD -- the ordering escapes the failure by abandoning the bound or the narrowing, turning a
  30-60s-per-attempt GPU gate into a grid walk, or stepping past a genuine cheap-corner defect and
  hiding real evidence.

Same contract and hazards as `revert_check_s2.py` (see its `run()` docstring for why `-B` and `-v`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "src" / "kernel_optimizer" / "paramspace" / "validation.py"
TESTS = [ROOT / "tests" / "test_witness_fallback_order.py"]

_ORDERED = ('        ordered = sorted(itertools.product(*[d.choices for d in space.domains]),\n'
            '                         key=distance_from_failures)')

VARIANTS: list[tuple[str, Path, str, str, list[str], str]] = [
    (
        "the PRE-FIX walk: plain itertools.product in declaration order",
        VAL,
        _ORDERED,
        "        ordered = itertools.product(*[d.choices for d in space.domains])",
        ["test_the_retries_do_not_reuse_the_failed_precision",
         "test_each_retry_differs_from_the_failures_in_more_than_one_knob",
         "test_it_finds_a_witness_when_only_the_failed_precision_is_bad"],
        "the exact state that produced the rejection this fix came from. `product` varies its LAST "
        "factor fastest, so on level3/48's real 7-knob space both retries came back "
        "COMPUTE_DTYPE=fp16 -- the overflowing dtype -- differing from the dead witness only in "
        "NUM_STAGES 1->2 and 1->3. The fallback spent its whole budget re-confirming the failure, "
        "and the candidate was rejected AFTER repair had already fixed its algorithm",
    ),
    (
        "distance measured against the DEFAULT only, not every failed config",
        VAL,
        "            nearest = min(sum(1 for k in names if vals.get(k) != t.get(k)) for t in tried)\n"
        "            return (-nearest, 0)",
        "            first = tried[0]\n"
        "            return (-sum(1 for k in names if vals.get(k) != first.get(k)), 0)",
        ["test_the_retries_do_not_reuse_the_failed_precision"],
        "the plausible half-fix. `tried` holds BOTH dead configs and the minimal one is the "
        "overflow; maximising distance from the default alone lands on fp16 in both retries, "
        "because being far from `ieee` is exactly what fp16 is. Using min() over all failures is "
        "what makes the ordering avoid the actual failure rather than one arbitrary point. NOTE my "
        "first version of this variant also named the `more_than_one_knob` test, which correctly "
        "still PASSED: these retries sit 5 knobs from the minimal witness, so they are genuine "
        "departures that happen to reuse the dead dtype. Distance and dtype-reuse are two "
        "different claims and only one of them is what this variant breaks",
    ),
    (
        "the sort inverted: retries closest to the failures first",
        VAL,
        "            return (-nearest, 0)",
        "            return (nearest, 0)",
        ["test_the_retries_do_not_reuse_the_failed_precision",
         "test_each_retry_differs_from_the_failures_in_more_than_one_knob"],
        "a sign error, which is the single most likely way for this to break silently later. "
        "It is strictly WORSE than the pre-fix state: it deliberately picks the nearest "
        "neighbours of the dead config, i.e. it optimises for re-confirming the overflow",
    ),
    (
        "the retry budget removed while keeping the better ordering",
        VAL,
        "            if attempted >= self.max_witness_retries:\n                return None",
        "            if False:\n                return None",
        ["test_the_retry_budget_is_still_respected"],
        "trades a rejected candidate for an unbounded walk. Each attempt is a real GPU quick test "
        "(30-60s on L3) and these spaces run to thousands of configs, so a space where nothing "
        "passes would burn hours inside a gate whose job is to decide in about a minute. The "
        "reordering is worth having only because the bound stays",
    ),
    (
        "already-failed configs allowed back into the walk",
        VAL,
        "            if params.values in tried:\n                continue",
        "            if False:\n                continue",
        [],
        "UNDISCRIMINATING BY CONSTRUCTION, and left here as the evidence of why. With the distance "
        "ordering a dead config scores distance 0 -- the minimum -- so it sorts LAST and is never "
        "reached inside the 2-attempt budget. Verified: this variant still picks bf16 twice, "
        "identical to the fixed version. The `in tried` check is therefore defence in depth for "
        "the day the ordering changes again, not a load-bearing guard today; naming any test here "
        "would be claiming evidence the harness cannot provide. "
        "`test_an_already_failed_config_is_never_retested` earns its place against the PRE-FIX "
        "ordering, where dead configs sort early -- and the first variant above is what covers it",
    ),
    (
        "the out-of-range narrowing dropped: fall back on ANY minimal failure",
        VAL,
        "                if label == \"minimal\" and _looks_out_of_range(result):",
        "                if label == \"minimal\":",
        [],
        "the TOO-BROAD direction, and the one with a real cost: level3/21 and level3/43's "
        "historical minimal-witness failures carry zero non-finite values and max-abs-diff of "
        "0.0013-0.0040 on bounded outputs -- plausibly a genuine defect that only shows at the "
        "cheap corner, and stepping past those hides real evidence. UNDISCRIMINATING HERE, "
        "correctly: every test in this file exercises `_next_witness` or `_looks_out_of_range` "
        "directly, and this variant changes neither -- it changes the CALL SITE that decides "
        "whether to invoke the fallback. Covering it behaviourally needs a full "
        "`validate_and_publish` run against a fake worker, which is `test_space_validation.py`'s "
        "territory, not this file's. Named with no tests rather than with a test that would pass "
        "for an unrelated reason",
    ),
    (
        "max() instead of min() over the failed configs",
        VAL,
        "            nearest = min(sum(1 for k in names if vals.get(k) != t.get(k)) for t in tried)",
        "            nearest = max(sum(1 for k in names if vals.get(k) != t.get(k)) for t in tried)",
        ["test_the_retries_do_not_reuse_the_failed_precision"],
        "the one-character slip, and the sharpest illustration of why the aggregation is min(). "
        "max() asks 'how far is this from the FURTHEST failure', which a config sitting right on "
        "top of another failure can score highly -- measured: both retries come back fp16, the "
        "overflowing dtype, at distance 5 from the minimal witness and 7 from the default. min() "
        "asks 'how far from the NEAREST failure', which is the question that matters. NOTE this "
        "slot cost me three wrong guesses, each of which escaped the failure for an unrelated "
        "reason and so did not discriminate: `reverse=True` on the already-negated key (picks "
        "bf16/tf32), a bare `sorted()` with no key (lexicographic puts \"bf16\" before \"fp16\", an "
        "alphabetical accident), and counting knobs that compare LOWER rather than differ. The "
        "candidates were finally enumerated and MEASURED rather than reasoned about",
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
    originals = {p: p.read_text(encoding="utf-8") for p in (VAL,)}
    ok = True
    no_evidence = 0
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

            # A variant that does not parse fails every test for the WRONG reason, which reads as
            # discrimination (G43).
            patched = text.replace(old, new)
            try:
                compile(patched, str(path), "exec")
            except SyntaxError as exc:
                print("**SKIPPED** %s: the patched file does not parse (%s), so any failure it "
                      "produced would be for the wrong reason" % (label, exc))
                ok = False
                continue

            failed, was_skipped, out = apply_variant(path, text, old, new, must_fail)
            wrongly_passed = [n for n in must_fail if n not in failed and n not in was_skipped]

            # A variant naming NO tests is a documented non-discrimination, not a pass. Two of them
            # here are undiscriminating by construction (see their `why`), and printing them as
            # "ok" would be exactly the false green this harness exists to prevent.
            if not must_fail:
                no_evidence += 1
                print("NOEVID  %s" % label)
                print("        names no tests: this variant is not covered by this file")
                print("        wrong version: %s" % why)
                print()
                continue

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
    if no_evidence:
        print("\nVERDICT: every variant this file covers discriminates; %d variant(s) named no "
              "tests and are documented as uncovered here (see NOEVID above)" % no_evidence)
        return 0
    print("\nVERDICT: every test fails on the implementation it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
