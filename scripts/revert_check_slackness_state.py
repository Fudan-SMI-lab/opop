"""Revert-check for `check_wrapup.check_s3_s4`'s complementary-slackness state.

THE DEFECT THIS GUARDS WAS FOUND IN THIS FILE'S OWN SUBJECT. `check_s3_s4` used to count occurrences
of four strings -- "cannot_run", "no_dimension_judged_slack", "weak_pass", "violation_named" -- anywhere
in an event payload. Not one of them appears anywhere in `src/`: the real check lives in
`evaluation/conversion_report.py` and keys on `DIMENSION_STATE.records[].verdict == "slack"` with
`applicable`. So the reader printed `none` on a fully working run, which at wrap-up reads as "S4'
produced no states". Six tests had passed for weeks without touching that line.

AND THE FIRST VERSION OF THIS HARNESS WAS ITSELF INVALID, which is why the reversion is anchored on
the whole derivation rather than on one line. Patching `if n_dimension_states == 0:` to `if False:`
left the surviving `elif` / `else` branches to recompute the correct answer, so the variant produced
IDENTICAL output to the fix and all six tests passed on it -- reported as "0 of 6 caught". A variant
that does not actually change behaviour is not evidence of anything; the fix is to remove the whole
block the tests are about.

    python scripts/revert_check_slackness_state.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])
TARGET = ROOT / "scripts" / "check_wrapup.py"
TESTS = "tests/test_check_wrapup.py"

# The unique start and end of the derivation. Both must appear exactly once, or the anchor has moved
# and the harness says so instead of silently testing nothing.
START = """    # The four states, DERIVED the way `conversion_lines` derives them, so this reports what the
    # report will say instead of a second opinion that can silently disagree."""
END = """                slack_state = ("weak_pass: no violation over %d round(s) with a verdict and %d "
                               "slack dimension(s) (%s)" % (
                                   len(rounds_with_conversion), len(slack_dims),
                                   ", ".join(sorted(slack_dims))))"""

# The pre-fix reader, verbatim in behaviour: four invented names counted over payload text.
ORIGINAL = '''    slack_states = collections.Counter()
    for _e in _events(run_dir):
        blob = json.dumps(_e.get("payload") or {})
        for state in ("cannot_run", "no_dimension_judged_slack", "weak_pass", "violation_named"):
            if state in blob:
                slack_states[state] += 1
    slack_state = str(dict(slack_states) or "none")'''

CASES = [
    ("no_dimension_judged_slack_is_reported_as_not_a_pass",
     "verdicts exist but none is slack: must say NOT a pass, not print `none`"),
    ("an_inapplicable_slack_dimension_does_not_count",
     "`applicable` is load-bearing -- a dimension with no polarity is not slack in the "
     "shadow-price sense, and counting it manufactures violations"),
    ("slack_dimensions_with_no_verdict_bearing_round_cannot_run",
     "slack dimensions with no round to cross them against is `cannot_run`, not a pass"),
    ("a_slack_dimension_improving_with_a_latency_gain_is_named_a_violation",
     "the whole point: a slack dimension improving alongside a latency gain is evidence OUR "
     "verdict was wrong, and it must be named"),
    ("no_violation_over_real_rounds_is_a_weak_pass_not_a_pass",
     "no violation over one round is not the same claim as `the verdicts are right`"),
    ("the_state_is_never_one_of_four_invented_labels",
     "the same log must yield different states depending on the RECORDS -- which a substring "
     "counter over payload text cannot do"),
]


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    if src.count(START) != 1 or src.count(END) != 1:
        print("ANCHOR MOVED -- start x%d, end x%d. Re-anchor before trusting any result below."
              % (src.count(START), src.count(END)))
        return 1
    i, j = src.index(START), src.index(END) + len(END)
    variant = src[:i] + ORIGINAL + src[j:]
    io.open(TARGET, "w", encoding="utf-8", newline="\n").write(variant)
    try:
        print("VARIANT: the pre-fix reader (four names counted over payload text)")
        caught = []
        for name, why in CASES:
            r = subprocess.run(
                [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                 "-p", "no:cacheprovider", "-k", name],
                cwd=ROOT, capture_output=True,
                env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
            ok = r.returncode != 0
            print("  %-58s %s" % (name[:58], "FAILS as required" if ok else "*** STILL PASSES ***"))
            print("      %s" % why)
            if ok:
                caught.append(name)
    finally:
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src)
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True,
                       env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    tail = r.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
    print("\nrestored: %s" % tail)
    if len(caught) != len(CASES):
        print("%d of %d cases do not fail on the pre-fix reader." % (
            len(CASES) - len(caught), len(CASES)))
        return 1
    print("Every case fails on the reader it was written against.")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
