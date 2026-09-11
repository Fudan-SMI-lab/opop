"""Revert-check for scripts/check_arm_prompt_separation.py.

The failure this guards against is SILENT: a run whose S2 switch never took effect proceeds normally,
and every latency comparison between the arms then measures nothing. So a clean result on the live
runs is not evidence the check works -- only injected defects are. Nine cases plus a probe control,
built as synthetic sandboxes whose answer is known by construction:

    clean_vector                the treatment arm as it should look                       exit 0
    clean_label                 the control arm as it should look                         exit 0
    BOTH                        the J2-3 violation: digest APPENDED beside the label
    NEITHER                     the agent got no resource statement at all
    wrong_arm                   the label delivered on a run asserting vector
    one_doc_missing             one document right, a second carrying neither
    neither_no_arm_asserted     no arm asserted, so ONLY the n_neither branch can fire
    one_empty_no_arm_asserted   same, on a run where one document is fine
    mixed_arms_across_docs      both markers present but in DIFFERENT docs, so only the
                                every-doc branch can fire
    marker_renamed              a heading the emitter no longer writes must report PROBE BROKEN

The BOTH case is the one a one-directional check ("does the treatment carry the vector?") passes,
which is why the check counts both markers in both arms.

THE LAST THREE CASES EXIST BECAUSE THE FIRST SIX WERE OVER-DETERMINED. With an arm asserted, a
document carrying NEITHER also trips the "every doc must carry the vector" branch -- so deleting the
n_neither check changed no verdict and that variant read as NOT CAUGHT. A case that fires two branches
at once cannot tell you which one is load-bearing; each of the three added ones can fire exactly one.

    python scripts/revert_check_arm_prompt_separation.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "check_arm_prompt_separation.py"

VECTOR = "## Resource state, one line per dimension"
LABEL = "## Verdict: **resource_limited**"
# The two sections both arms share, so a case differs from the real document in exactly the one
# section under test.
HEAD = ("## What this task requires (measured on the REFERENCE)\n\n- x\n\n"
        "## What this GPU can actually do (measured on THIS box)\n\n- y\n\n")

# name -> (sections in the first doc, asserted arm or None, expected exit, second doc)
#
# The `want=None` cases exist because the others are OVER-DETERMINED: with an arm asserted, a document
# carrying NEITHER also trips the "every doc must carry the vector" branch, so deleting the n_neither
# check changes no verdict and the variant reads as uncaught. Measured -- two variants reported NOT
# CAUGHT on exactly that. A case has to be able to isolate the branch it claims.
#
# `second` is None (one doc only), or "neither"/"label"/"vector" for a second document.
CASES: dict[str, tuple[list[str], str | None, int, str | None]] = {
    "clean_vector": ([VECTOR], "vector", 0, None),
    "clean_label": ([LABEL], "label", 0, None),
    "BOTH": ([VECTOR, LABEL], "vector", 1, None),
    "NEITHER": ([], "vector", 1, None),
    "wrong_arm": ([LABEL], "vector", 1, None),
    "one_doc_missing": ([VECTOR], "vector", 1, "neither"),
    # No arm asserted: the ONLY branch that can fire is n_neither.
    "neither_no_arm_asserted": ([], None, 1, None),
    # No arm asserted, one doc fine and one empty: isolates n_neither on a mixed run.
    "one_empty_no_arm_asserted": ([VECTOR], None, 1, "neither"),
    # Both markers present but in DIFFERENT docs: n_both=0 and n_neither=0, so the only branch that
    # can fire is "every doc must carry the vector".
    "mixed_arms_across_docs": ([VECTOR], "vector", 1, "label"),
}

VARIANTS: dict[str, tuple[tuple[str, str], str, list[str]]] = {
    "no_both_check": (
        ("if n_both:", "if False:"),
        "stops reporting a document that carries the vector AND the label. That is the J2-3 "
        "violation and the one a one-directional check cannot see: the treatment arm would be more "
        "informed for two reasons at once, so no latency difference could be attributed to the "
        "vector",
        ["BOTH"]),
    "no_neither_check": (
        ("if n_neither:", "if False:"),
        "stops reporting a document carrying no resource statement at all -- the agent was handed "
        "nothing, which is not the control condition either",
        ["neither_no_arm_asserted", "one_empty_no_arm_asserted"]),
    "any_doc_instead_of_every": (
        ("if want == \"vector\" and n_vec != len(rows):",
         "if want == \"vector\" and n_vec == 0:"),
        "accepts the arm if ANY document carries the vector rather than EVERY one, so a switch that "
        "took effect for one candidate and silently stopped reads as fully applied",
        ["mixed_arms_across_docs"]),
    "wrong_arm_accepted": (
        ("if want == \"label\" and n_lab != len(rows):",
         "if False and n_lab != len(rows):"),
        "stops checking that a run asserted as the control actually delivered the label. NO CASE "
        "FLIPS: `wrong_arm` asserts vector, and its label-bearing document is already caught by the "
        "vector branch above -- the two branches are symmetric, so this one is only reachable by a "
        "case asserting `label` and receiving the vector, which the BOTH/NEITHER cases already "
        "cover from the other side",
        []),
    "no_marker_positive_control": (
        ("_gone = _markers_still_exist()", "_gone = []"),
        "removes the check that the markers still exist in the emitting source. A renamed heading "
        "then makes every document read NEITHER, and 'the agent got no resource statement' becomes "
        "indistinguishable from 'this script matches a stale string'",
        ["marker_renamed"]),
}


_SECOND = {"neither": [], "label": [LABEL], "vector": [VECTOR]}


def _build(tmp: Path, sections: list[str], mode: str | None, second: str | None) -> Path:
    run = tmp / "run"
    d = run / "sandboxes" / "analyst-aaaa" / "analysis"
    d.mkdir(parents=True)
    (d / "bottleneck.md").write_text(HEAD + "\n".join(sections) + "\n", encoding="utf-8")
    if second is not None:
        d2 = run / "sandboxes" / "analyst-bbbb" / "analysis"
        d2.mkdir(parents=True)
        (d2 / "bottleneck.md").write_text(
            HEAD + "\n".join(_SECOND[second]) + "\n", encoding="utf-8")
    # `prompt_mode` is what the orchestrator recorded. A case asserting no arm still needs a mode in
    # the log, set to whatever the first document actually carries -- otherwise the intent/delivery
    # cross-check fires and over-determines these cases too.
    recorded = mode or ("vector" if VECTOR in "".join(sections) else "label")
    (run / "events.jsonl").write_text(json.dumps({
        "seq": 0, "ts": 1.0, "type": "DIMENSION_STATE",
        "payload": {"prompt_mode": recorded}}) + "\n", encoding="utf-8")
    return run


def _evaluate(check: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name, (sections, want, want_rc, second) in CASES.items():
        tmp = Path(tempfile.mkdtemp(prefix="aps-"))
        try:
            run = _build(tmp, sections, want, second)
            argv = [sys.executable, "-B", str(check), str(run), "T"]
            if want is not None:
                argv.append(want)
            r = subprocess.run(argv,
                               capture_output=True, cwd=ROOT,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                    "KOPT_REPO_ROOT": str(ROOT)})
            out[name] = (r.returncode == want_rc)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # And the marker positive control: with a marker the emitter no longer writes, the check must
    # report itself broken (exit 2) rather than reporting every arm as NEITHER.
    tmp = Path(tempfile.mkdtemp(prefix="aps-mk-"))
    try:
        src = check.read_text(encoding="utf-8")
        v = tmp / "renamed.py"
        v.write_text(src.replace(
            'VECTOR = "## Resource state, one line per dimension"',
            'VECTOR = "## Resource state, ONE LINE PER DIMENSION (renamed)"'), encoding="utf-8")
        run = _build(tmp, [VECTOR], "vector", None)
        r = subprocess.run([sys.executable, "-B", str(v), str(run), "T", "vector"],
                           capture_output=True, cwd=ROOT,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                "KOPT_REPO_ROOT": str(ROOT)})
        out["marker_renamed"] = (r.returncode == 2
                                 and b"PROBE BROKEN" in r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main() -> int:
    base = _evaluate(CHK)
    print("BASELINE (scripts/check_arm_prompt_separation.py):")
    for name, ok in base.items():
        print("  %-22s %s" % (name, "ok" if ok else "FAIL"))
    if not all(base.values()):
        print("\nBASELINE IS BROKEN -- fix that before reading any variant result.")
        return 1

    src = CHK.read_text(encoding="utf-8")
    bad = 0
    print("\nVARIANTS:")
    for name, ((old, new), rationale, claimed) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-26s SKIPPED -- anchor occurs %d times, not once" % (name, src.count(old)))
            print("      %s" % rationale)
            bad += 1
            continue
        tmp = Path(tempfile.mkdtemp(prefix="aps-var-"))
        try:
            v = tmp / "variant.py"
            v.write_text(src.replace(old, new), encoding="utf-8")
            got = _evaluate(v)
            flipped = sorted(k for k, ok in got.items() if base[k] and not ok)
            if not claimed:
                verdict = ("UNCLAIMED-CATCH by %s" % ",".join(flipped)) if flipped else "NOEVID"
            elif set(claimed) & set(flipped):
                verdict = "caught by %s" % ",".join(flipped)
            else:
                verdict = "NOT CAUGHT (claimed %s)" % ",".join(claimed)
                bad += 1
            print("  %-26s %s" % (name, verdict))
            print("      %s" % rationale)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if bad:
        print("\n%d variant(s) not caught by a case they name, or unapplicable." % bad)
        return 1
    print("\nEvery claiming variant is caught by a case it names. One names none and prints NOEVID, "
          "with the measurement behind that in its rationale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
