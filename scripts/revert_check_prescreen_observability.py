"""Revert-check for the prescreen's cost fields (`answered` / `elapsed_s` / `timed_out`).

The failure this defends against is INVISIBILITY, not a wrong number. Box 3's two runaway batches sat
at 1200.5 s and 1201.0 s -- exactly `build_timeout_s` -- and each reported `infeasible: 0`, which is
byte-identical to a fast batch that legitimately found nothing to reject. I only caught it by noticing
the GPU was at 0%, which is not a method.

So the variants here are all "the field goes away or stops distinguishing", and the test that catches
each is the one asserting the distinction rather than the value.

MUST RUN ON THE A800. `tests/test_prescreen_observability.py` imports `CandidateRun`, which pulls in
optuna, which Windows does not have -- so on Windows every variant reads UNVERIFIED, and a skip is not
a verdict. This project has twice had a Windows skip hide a real failure from a revert-check.

    PYTHONPATH=src python scripts/revert_check_prescreen_observability.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])
TARGET = ROOT / "src" / "kernel_optimizer" / "control" / "orchestrator.py"
TESTS = "tests/test_prescreen_observability.py"

VARIANTS = {
    "answered_never_counted": (
        ('            "answered": answered, "elapsed_s": round(elapsed, 1),\n'
         '            "timed_out": answered == 0 and elapsed >= 0.9 * '
         'self.cfg.evaluation.build_timeout_s,\n',
         ""),
        "drops all three fields, i.e. exactly the state box 3 ran in: a 1200 s batch that answered "
        "nothing is indistinguishable in the log from a 0.4 s batch that found nothing, and the only "
        "way to notice is to catch the GPU idle",
        ["a_prescreen_that_answers_nothing_is_distinguishable_from_one_that_finds_nothing"]),
    "answered_aliases_infeasible": (
        ("        answered = sum(1 for v in verdicts if v is not None)",
         "        answered = infeasible"),
        "makes `answered` a second name for `infeasible`. Then a batch that answered all 40 FEASIBLE "
        "reports answered=0 and reads as a timeout -- the false positive that would make the field "
        "useless, since most batches legitimately find nothing to reject",
        ["a_fast_batch_that_finds_nothing_is_NOT_flagged_as_a_timeout",
         "answered_is_not_the_same_count_as_infeasible"]),
    "timeout_keyed_on_elapsed_alone": (
        ('            "timed_out": answered == 0 and elapsed >= 0.9 * '
         'self.cfg.evaluation.build_timeout_s,',
         '            "timed_out": elapsed >= 0.9 * self.cfg.evaluation.build_timeout_s,'),
        "flags any slow batch as a timeout, including one that produced every answer before the "
        "deadline. Expensive and uninformative are different states and only one of them is a defect",
        ["a_slow_batch_that_DID_answer_is_not_a_timeout"]),
    "infeasible_counts_every_answer": (
        ("        infeasible = sum(1 for v in verdicts if v is False)",
         "        infeasible = sum(1 for v in verdicts if v is not None)"),
        "counts a FEASIBLE verdict as infeasible. Silent and in the direction that matters: it would "
        "report configurations as pre-rejected that the screen actually cleared, inflating the "
        "screen's apparent payoff exactly where I was trying to measure whether it earns its cost",
        ["answered_is_not_the_same_count_as_infeasible"]),
}


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    base = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True,
        env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    tail = base.stdout.decode("utf-8", "replace").strip().splitlines()
    print("baseline: %s" % (tail[-1] if tail else "no output"))
    if base.returncode != 0:
        print("BASELINE IS NOT GREEN -- fix that before trusting any variant below.")
        print("  (on Windows this is expected: the tests import CandidateRun -> optuna. "
              "Run this on the A800.)")
        return 1

    bad = 0
    for name, ((old, new), why, claimed) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-32s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            flipped, skipped = [], []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t],
                    cwd=ROOT, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                out = r.stdout.decode("utf-8", "replace")
                # A SKIP is not a verdict, so it must be told apart from a real pass. Written as a
                # plain statement rather than a conditional expression: the first version was
                # `if A or B if out else False`, whose precedence makes the whole test the ternary's
                # condition -- it would have reported skips as passes.
                last = out.strip().splitlines()[-1] if out.strip() else ""
                was_skipped = ("no tests ran" in out) or ("skipped" in last and "passed" not in last)
                if was_skipped:
                    skipped.append(t)
                elif r.returncode != 0:
                    flipped.append(t)
            if skipped and not flipped:
                print("  %-32s UNVERIFIED -- the naming test(s) SKIPPED, which is not a verdict: %s"
                      % (name, ", ".join(skipped)))
                bad += 1
            elif flipped:
                print("  %-32s CAUGHT by %s" % (name, ", ".join(x[:52] for x in flipped)))
            else:
                print("  %-32s *** NOT CAUGHT *** (claimed %s)" % (name, claimed))
                bad += 1
            print("      %s" % why)
        finally:
            io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src)

    r = subprocess.run([sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True,
                       env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    out = r.stdout.decode("utf-8", "replace").strip().splitlines()
    print("\nrestored: %s" % (out[-1] if out else "no output"))
    if bad:
        print("%d variant(s) skipped, unverified, or not caught." % bad)
        return 1
    print("Every variant is caught by a case it names.")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
