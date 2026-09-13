"""Would the new TOTAL-SEARCH check's tests actually fail without it?

WHY. Six tests were added for a check that fires on the live S7 pair. Tests written after a fix have a
way of passing for reasons unrelated to it -- the recorded `a-clean-run-is-not-evidence-a-checker-works`
and `a-variant-that-changes-no-behaviour-is-not-a-variant` shapes. So each variant below breaks the new
check in a DIFFERENT plausible way and the suite must fail on every one.

The variants are chosen to be things I might genuinely have written:
  1. no check at all               -- the state of the file before this change
  2. a fixed percentage tolerance  -- the obvious alternative to deriving the unit from the data
  3. judge it mid-run too          -- forget the RUN_FINISHED gate
  4. compare spaces, not trials    -- count spaces instead of the search they bought
  5. drop the mechanism sentence   -- report totals without naming K expansion
  6. use max instead of the mode   -- pick a unit that a 41-trial space perturbs
  7. abstain silently when unknown -- also silence the AGREEING branch, so nothing is ever reported
  8. raw trials against the unit   -- THE FIRST DRAFT OF THIS CHECK, which this probe caught: a space
                                      truncated to 1 trial reads as a 39-trial gap against a unit of 40

Variants 2, 4 and 6 were NOT CAUGHT on the first run of this probe, and the fault was in the tests, not
the check: every fixture had space count, trial count and modal budget moving together, so the
alternatives agreed with the shipped code on all of them. Variant 8 then found a real defect in the
check itself. Three tests were added to pull those quantities apart -- which is the whole reason to
write a revert check rather than trust a green suite.

Run:  PYTHONIOENCODING=utf-8 uv run --offline --extra test --quiet python scripts/probes/revert_check_total_search.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path("scripts/check_arm_search_parity.py")
TESTS = Path("tests/test_arm_search_parity.py")

# The block added by this change, matched by its first and last line so a reformat does not silently
# turn a variant into a no-op.
_START = '    unit = ma if (ma is not None and ma == mb) else None'
_END = '                     % (ta, tb, sa, sb, gap_spaces, _SPACE_TOL, ea, eb))'


def _block(text: str) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    i = next(n for n, ln in enumerate(lines) if ln.rstrip("\n") == _START)
    j = next(n for n, ln in enumerate(lines) if ln.rstrip("\n") == _END)
    return i, j


VARIANTS: dict[str, "callable"] = {}


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn
    return deco


@variant("1-no-check-at-all")
def _v1(text: str) -> str:
    """The file as it was: collect `expansions`, print it, never judge on it."""
    lines = text.splitlines(keepends=True)
    i, j = _block(text)
    return "".join(lines[:i] + lines[j + 1:])


@variant("2-fixed-percent-tolerance")
def _v2(text: str) -> str:
    """A 25% relative tolerance instead of one space-equivalent. Discriminated by the long-run test:
    240 vs 200 trials is a whole extra space at only 20%, so a percentage silently stops noticing an
    extra space as the run gets longer."""
    return text.replace(
        "    if unit and gap_spaces >= _SPACE_TOL:",
        "    if unit and min(ta, tb) and abs(ta - tb) / min(ta, tb) >= 0.25:")


@variant("3-judged-mid-run")
def _v3(text: str) -> str:
    """Forget that mid-run arms are out of step by construction."""
    return text.replace(
        """        if live:
            notes.append("PROVISIONAL, " + msg + " Mid-run the trailing arm may still catch up, so "
                         "this is recorded rather than judged; re-run at the end.")
        else:
            okay = False
            notes.append(msg)""",
        """        okay = False
        notes.append(msg)""")


@variant("4-compare-space-counts")
def _v4(text: str) -> str:
    """Count spaces rather than the search they bought. Discriminated by the equal-space-count test:
    a space truncated by the wall clock bought 1 trial, not 40, and space counting sees parity."""
    return text.replace(
        "    if unit and gap_spaces >= _SPACE_TOL:",
        "    if unit and abs(sa - sb) >= 1:")


@variant("5-no-mechanism-named")
def _v5(text: str) -> str:
    """Report the totals but not WHY they differ. A reader who cannot see 'K expansion' cannot tell an
    expansion asymmetry from an arm that simply died early."""
    lines = []
    for ln in text.splitlines(keepends=True):
        if '" %s made %d K expansion(s) against %d, and each expansion publishes another space "' in ln:
            continue
        if '"over the same candidate => another %d trials of budget" %' in ln:
            continue
        if '(la if ea > eb else lb, max(ea, eb), min(ea, eb), unit))' in ln:
            lines.append('            mech = ""\n')
            continue
        lines.append(ln)
    return "".join(lines).replace("            mech = (\n", "")


@variant("6-max-instead-of-mode")
def _v6(text: str) -> str:
    """Derive the unit from the largest closed space rather than the modal one. Discriminated by the
    overshoot test: one 41-trial space then prices a 40-trial gap at 0.98 spaces and it goes silent."""
    return text.replace(
        "    unit = ma if (ma is not None and ma == mb) else None",
        "    unit = max(ca + cb) if (ca and cb) else None")


@variant("8-exact-unit-not-space-equivalents")
def _v8(text: str) -> str:
    """The first draft of this very check: compare the raw trial difference against `unit`. A space
    truncated to 1 trial then reads as a 39-trial gap against a unit of 40 and passes -- the arm that
    lost a whole space to the wall clock is the case it misses."""
    return text.replace(
        "    if unit and gap_spaces >= _SPACE_TOL:",
        "    if unit and abs(ta - tb) >= unit:")


@variant("7-silent-when-unknown")
def _v7(text: str) -> str:
    """Abstain when there is no modal budget -- correct -- but also drop the AGREEING branch, so a
    healthy pair gets no total-search line at all and 'nothing was reported' becomes indistinguishable
    from 'nothing was wrong'."""
    lines = text.splitlines(keepends=True)
    out = []
    skip = False
    for ln in lines:
        if ln.rstrip("\n") == "    elif unit:":
            skip = True
            continue
        if skip:
            if ln.rstrip("\n") == _END:
                skip = False
            continue
        out.append(ln)
    return "".join(out)


def main() -> int:
    original = SRC.read_text(encoding="utf-8")
    # Sanity: the block must be locatable, or every variant below is a no-op that "passes".
    _block(original)

    caught = 0
    for name, fn in VARIANTS.items():
        mutated = fn(original)
        if mutated == original:
            print("%-26s NO-OP VARIANT -- it changed nothing, so it proves nothing" % name)
            continue
        SRC.write_text(mutated, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-x"],
                capture_output=True)
            out = r.stdout.decode("utf-8", "replace")
            failed = r.returncode != 0
            m = re.search(r"(\d+) failed", out)
            which = re.findall(r"^FAILED (\S+)", out, re.M) or re.findall(
                r"^(tests/\S+::\w+)", out, re.M)
            if failed:
                detail = "%s failing: %s" % (
                    m.group(1) if m else "?",
                    ", ".join(w.split("::")[-1] for w in which[:3]))
            else:
                tail = out.strip().splitlines()
                detail = tail[-1] if tail else "(no output)"
            print("%-26s %s   %s" % (
                name, "CAUGHT" if failed else "*** NOT CAUGHT ***", detail))
            caught += bool(failed)
        finally:
            SRC.write_text(original, encoding="utf-8")

    print()
    print("%d/%d variants caught." % (caught, len(VARIANTS)))
    return 0 if caught == len(VARIANTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
