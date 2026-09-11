"""Revert-check for `check_wrapup.check_arm_parity`.

The premise of every paired number this checker prints is "one thing changed between the two runs". If
that premise is false the comparison is not weakened, it answers a different question -- and the
failure is silent, because both runs finish normally and produce comparable-looking latencies.

Found by measurement, not by suspicion: the live pair differs in a THIRD config key (`wsl.venv`), and
establishing it was benign took a manual check of both interpreters on both boxes. That is now the
conditional path exemption, keyed on the recorded calibration identity.

EACH VARIANT IS VERIFIED TO CHANGE BEHAVIOUR BEFORE ITS TESTS ARE RUN. An earlier harness in this
project patched an `if` condition to `False` and left the surviving `elif`/`else` chain to recompute
the same answer -- the variant was byte-identical to the fix and all six tests "passed on the defect",
which read as the tests being worthless. A variant that changes nothing is not evidence of anything,
so this harness asserts a behavioural difference on a probe case first and reports SHAM if there is
none.

    python scripts/revert_check_arm_parity.py
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])
TARGET = ROOT / "scripts" / "check_wrapup.py"
TESTS = "tests/test_check_wrapup.py"

VARIANTS = {
    "path_exemption_granted_outright": (
        ('                  if k not in _ARM_SWITCHES and not (k in _ARM_PATHLIKE and same_env)}',
         '                  if k not in _ARM_SWITCHES and k not in _ARM_PATHLIKE}'),
        "exempts a differing venv path WITHOUT checking that both arms recorded the same "
        "calibration identity. torch/triton is a recorded covarying factor (G23), so this hides a "
        "real environment divergence behind a path that merely looks like a path",
        ["a_differing_venv_with_a_DIFFERENT_identity_is_a_confound"], "confound"),
    "no_same_arm_check": (
        ('    missing_switch = [k for k in _ARM_SWITCHES if k not in differing]',
         '    missing_switch = []'),
        "stops noticing that the two runs are the SAME arm. This is the failure the config layer's "
        "extra=forbid fix exists to prevent, at the run level: a v3 typo left both arms as control, "
        "the runs agreed, and the conclusion would have been `the vector changes nothing`",
        ["two_arms_that_are_the_SAME_arm_are_reported_as_a_defect",
         "only_one_switch_flipped_is_also_reported_as_a_defect"], "same_arm"),
    "unexpected_differences_ignored": (
        ('    elif unexpected:', '    elif False:'),
        "reports PASS even when keys differ beyond the switches -- a different trial budget, a "
        "different wall clock, a different model. Every one of those makes the latency comparison "
        "answer a different question while looking identical in the output",
        ["a_differing_budget_is_a_confound_even_with_one_environment"], "confound"),
    "absent_manifest_reads_as_parity": (
        ('    if not c or not t:', '    if False:'),
        "treats an unreadable manifest as agreement. Absence of a manifest is not evidence of "
        "parity, and this is the friendly-looking failure: it prints a verdict instead of refusing",
        ["a_missing_manifest_refuses_to_decide"], "no_manifest"),
}


# Each variant needs a probe pair that REACHES the branch it patches. One fixed pair cannot: only a
# same-arm pair reaches the missing-switch branch, and only a missing manifest reaches the
# refuse-to-decide branch -- so a single-pair probe reports SHAM on perfectly real variants. Measured:
# it did, on two of these four, which is the SHAM detector working on the harness rather than on the
# subject.
PROBES = {
    # name -> ((mode, ledger, venv, identity, write_manifest) for control, then treatment)
    "confound": (("label", False, "/root/a", "GPU|2.13.0", True),
                 ("vector", True, "/root/b", "GPU|2.9.1", True)),
    "same_arm": (("label", False, "/root/a", "GPU|2.13.0", True),
                 ("label", False, "/root/a", "GPU|2.13.0", True)),
    "no_manifest": (("label", False, "/root/a", "GPU|2.13.0", True),
                    ("vector", True, "/root/a", "GPU|2.13.0", False)),
}


def _probe(check_dir: Path, kind: str = "confound") -> str:
    """`check_arm_parity`'s verdict on one probe pair, so a variant that changes no behaviour can be
    told apart from a test that does not cover it."""
    tmp = Path(tempfile.mkdtemp(prefix="ap-probe-"))
    try:
        for name, (mode, led, venv, ident, has_man) in zip(("ctl", "trt"), PROBES[kind]):
            d = tmp / name
            d.mkdir(parents=True)
            if has_man:
                (d / "manifest.json").write_text(json.dumps({"config": {
                    "wsl": {"venv": venv},
                    "v3": {"diagnosis": {"mode": mode, "expectation_ledger": led}}}}),
                    encoding="utf-8")
            (d / "events.jsonl").write_text(json.dumps({
                "seq": 0, "ts": 1.0, "type": "DIMENSION_STATE",
                "payload": {"records": [],
                            "compute_ceiling_provenance": {"calibration_identity": ident}}}) + "\n",
                encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; sys.path.insert(0, 'scripts'); import check_wrapup; "
             "from pathlib import Path; "
             "print(check_wrapup.check_arm_parity(Path(sys.argv[1]), Path(sys.argv[2]))['verdict'])",
             str(tmp / "ctl"), str(tmp / "trt")],
            cwd=check_dir, capture_output=True,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
        return (r.stdout.decode("utf-8", "replace").strip()
                or r.stderr.decode("utf-8", "replace").strip()[-200:])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    baselines = {k: _probe(ROOT, k) for k in PROBES}
    print("BASELINE probe verdicts (one per branch, so each variant is probed where it acts):")
    for _k, _v in baselines.items():
        print("  %-12s %s" % (_k, _v[:96]))
    print()
    bad = 0
    for name, ((old, new), why, claimed, kind) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-34s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            probe = _probe(ROOT, kind)
            if probe == baselines[kind]:
                # The recorded trap: no behavioural change means the variant proves nothing, and
                # "tests still pass" would be misread as the tests being worthless.
                print("  %-34s SHAM -- identical verdict to the baseline, so it patches nothing "
                      "reachable" % name)
                print("      %s" % why)
                bad += 1
                continue
            flipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t],
                    cwd=ROOT, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                if r.returncode != 0:
                    flipped.append(t)
            if flipped:
                print("  %-34s CAUGHT by %s" % (name, ", ".join(x[:46] for x in flipped)))
            else:
                print("  %-34s *** NOT CAUGHT *** (claimed %s)" % (name, claimed))
                bad += 1
            print("      %s" % why)
        finally:
            io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src)
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                        "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True,
                       env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
    print("\nrestored: %s" % r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    if bad:
        print("%d variant(s) sham, skipped, or not caught." % bad)
        return 1
    print("Every variant changes behaviour AND is caught by a case it names.")
    return 0 if r.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
