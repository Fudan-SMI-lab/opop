"""Revert-check for scripts/check_payload_paths.py.

The checker's job is to catch a reader that walks a payload key one level too shallow -- the shape of
four of the five real reader bugs this project has hit. So the control injects THOSE FOUR, verbatim,
plus one loop-shaped variant, and confirms each is reported. It also injects the two FALSE POSITIVES
that the first version of the checker actually produced on real readers, and confirms neither is.

Why a positive control and not a clean run against the repo: the first version caught 1 of the 4 real
bugs and reported the repo clean. A clean result on code you believe is clean is not evidence -- it
is the same trap as a fixture built from the reader's expectations. Only injected known defects can
tell a working checker from a silent one.

The shapes come from `tests/fixtures/payload_shapes.jsonl`, which is one event per type CAPTURED FROM
REAL RUNS (box 1, box 2 and two finished corpus runs) with every scalar leaf replaced by a
placeholder. Key structure is real and emitter-derived; no measurement, path or source text is
committed. It deliberately includes the LATE-RUN events -- CONVERGENCE_DECIDED, RUN_FINISHED,
FAMILY_ROUND_RECORDED -- because a live mid-flight run has none of them, and two of the real bugs
lived exactly there: a control built only on live arms skips them and passes.

    python scripts/revert_check_payload_paths.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "check_payload_paths.py"
FIXTURE = ROOT / "tests" / "fixtures" / "payload_shapes.jsonl"

# (reader source, the key that must appear in the message, must_be_reported)
CASES: dict[str, tuple[str, str, bool]] = {
    "real_bug_baseline_latency": (
        'def go(e):\n'
        '    if e["type"] == "BASELINE_DONE":\n'
        '        p = e["payload"]\n'
        '        lat = p.get("latency_ms") or {}\n'
        '        return lat\n',
        "latency_ms", True),
    "real_bug_trial_params": (
        'def go(e):\n'
        '    if e["type"] == "TRIAL_DONE":\n'
        '        p = e["payload"]\n'
        '        return p["params"]\n',
        "params", True),
    "real_bug_convergence_verdict": (
        'def go(e):\n'
        '    if e["type"] == "CONVERGENCE_DECIDED":\n'
        '        p = e["payload"]\n'
        '        return p.get("verdict")\n',
        "verdict", True),
    "real_bug_dimension_findings": (
        'def go(e):\n'
        '    if e["type"] == "DIMENSION_STATE":\n'
        '        p = e["payload"]\n'
        '        return p.get("findings")\n',
        "findings", True),
    "loop_shaped_guard": (
        'def go(events):\n'
        '    for e in events:\n'
        '        if e.get("type") != "TRIAL_DONE":\n'
        '            continue\n'
        '        p = e.get("payload") or {}\n'
        '        s = p.get("status")\n'
        '        print(s)\n',
        "status", True),
    # --- must NOT be reported: both measured on real readers in this repo ---
    "fp_or_fallback": (
        'def go(e):\n'
        '    if e["type"] == "TRIAL_DONE":\n'
        '        pay = e["payload"]\n'
        '        t = pay.get("trial") or {}\n'
        '        return pay.get("candidate_id") or t.get("candidate_id") or "?"\n',
        "candidate_id", False),
    "fp_negated_guard": (
        'def go(events):\n'
        '    for e in events:\n'
        '        if e["type"] not in ("SPACE_REJECTED", "TRIAL_DONE"):\n'
        '            continue\n'
        '        pl = e["payload"]\n'
        '        cid = pl.get("candidate_id")\n'
        '        print(cid)\n',
        "candidate_id", False),
    "fp_comment_describing_the_bug": (
        'def go(e):\n'
        '    if e["type"] == "CONVERGENCE_DECIDED":\n'
        '        p = e["payload"]\n'
        '        # Reading `p["verdict"]` was one level too shallow; it nests under `decision`.\n'
        '        return (p.get("decision") or {}).get("verdict")\n',
        "verdict", False),
}

# Variants of the CHECKER itself. Each must break at least one case it names.
VARIANTS: dict[str, tuple[tuple[str, str], str, list[str]]] = {
    "block_ends_at_any_return": (
        (r"                if re.match(r'\s*(continue|return|break)\s*$', lines[k]):",
         r"                if re.match(r'\s*(continue|return|break)\b', lines[k]):"),
        "ends a branch at ANY return, not a bare one. `return p[\"params\"]` is the branch's WORK, "
        "so this truncates the read away. MEASURED: this exact version missed 3 of the 4 real bugs "
        "and printed a clean result -- the reason this control exists",
        ["real_bug_trial_params", "real_bug_convergence_verdict",
         "real_bug_dimension_findings"]),
    "negated_guard_treated_as_branch": (
        ('        negated = ("not in" in lines[i]) or ("!=" in lines[i])',
         '        negated = False'),
        "treats `if type not in (A, B): continue` as opening a branch rather than selecting what "
        "SURVIVES, so the rest of the function is attributed to the wrong event",
        ["loop_shaped_guard"]),
    "no_or_fallback_exemption": (
        ("                                 % (re.escape(key), re.escape(key)), line):",
         "                                 % (re.escape(key), re.escape(key)), \"\"):"),
        "reports a read that ALREADY falls back to the nested location on the same line. That is "
        "defensive code getting the right answer, and flagging it trains the reader to ignore this "
        "script. (Anchored on the regex's ARGUMENT line rather than the `if re.search(` line, "
        "because the call is wrapped across two lines and a single-line anchor matched nothing -- "
        "which the harness reported as SKIPPED rather than passing.)",
        ["fp_or_fallback"]),
    "comments_read_as_code": (
        ('                code = line.split("#", 1)[0] if "#" in line else line',
         '                code = line'),
        "reads COMMENTS as code, so a reader documenting the bug it fixed is reported as still "
        "having it -- punishing exactly the code that recorded the lesson",
        ["fp_comment_describing_the_bug"]),
    "no_positive_control": (
        ("    if wrapped < 8 or len(top) < 5:", "    if False:"),
        "removes the floor on how much shape was learned, so a run directory with no usable events "
        "reports every reader clean instead of reporting itself broken",
        []),
}


def _run(check: Path, tree: Path, runs: list[str]) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-B", str(check), *runs, "--root", str(tree)],
                          capture_output=True, cwd=ROOT,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return proc.returncode, proc.stdout.decode("utf-8", "replace") + proc.stderr.decode(
        "utf-8", "replace")


def _evaluate(check: Path) -> dict[str, bool]:
    """For each case: did the checker do the right thing? Cases are run one at a time, so one
    case's hit cannot be mistaken for another's."""
    out: dict[str, bool] = {}
    tmp = Path(tempfile.mkdtemp(prefix="cpp-"))
    try:
        (tmp / "scripts").mkdir()
        (tmp / "tests").mkdir()
        run_dir = tmp / "run"
        run_dir.mkdir()
        shutil.copy(FIXTURE, run_dir / "events.jsonl")
        for name, (src, key, want) in CASES.items():
            f = tmp / "scripts" / ("reader_%s.py" % name)
            f.write_text(src, encoding="utf-8")
            try:
                rc, txt = _run(check, tmp, [str(run_dir)])
                reported = rc == 1 and name in txt and key in txt
                out[name] = (reported == want)
            finally:
                f.unlink()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _probe_broken_case(check: Path) -> bool:
    """A run directory with nothing usable must report PROBE BROKEN, not a clean result."""
    tmp = Path(tempfile.mkdtemp(prefix="cpp-pb-"))
    try:
        (tmp / "scripts").mkdir()
        (tmp / "tests").mkdir()
        rd = tmp / "run"
        rd.mkdir()
        (rd / "events.jsonl").write_text(
            '{"seq": 0, "ts": 1.0, "type": "STEP_DONE", "payload": {"step_key": "x"}}\n',
            encoding="utf-8")
        rc, txt = _run(check, tmp, [str(rd)])
        return rc == 2 and "PROBE BROKEN" in txt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    if not FIXTURE.exists():
        print("MISSING FIXTURE %s -- it holds the emitter-derived payload shapes this control "
              "checks against, and without it nothing below means anything." % FIXTURE)
        return 2

    base = _evaluate(CHK)
    base_pb = _probe_broken_case(CHK)
    print("BASELINE (scripts/check_payload_paths.py):")
    for name, ok in base.items():
        want = "must report" if CASES[name][2] else "must NOT report"
        print("  %-32s %-16s %s" % (name, want, "ok" if ok else "FAIL"))
    print("  %-32s %-16s %s" % ("probe_broken_on_empty_run", "must report",
                                "ok" if base_pb else "FAIL"))
    if not all(base.values()) or not base_pb:
        print("\nBASELINE IS BROKEN -- fix that before reading any variant result. On a broken "
              "baseline a variant 'caught' means nothing.")
        return 1

    src = CHK.read_text(encoding="utf-8")
    bad = 0
    print("\nVARIANTS:")
    for name, ((old, new), rationale, claimed) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-32s SKIPPED -- anchor occurs %d times, not once, so the variant would not "
                  "be the change it claims" % (name, src.count(old)))
            print("      %s" % rationale)
            bad += 1
            continue
        tmp = Path(tempfile.mkdtemp(prefix="cpp-var-"))
        try:
            v = tmp / "variant.py"
            v.write_text(src.replace(old, new), encoding="utf-8")
            got = _evaluate(v)
            got_pb = _probe_broken_case(v)
            flipped = sorted(k for k, ok in got.items() if base[k] and not ok)
            if not base_pb or not got_pb:
                flipped.append("probe_broken_on_empty_run")
            if not claimed:
                verdict = ("caught by %s" % ",".join(flipped)) if flipped else "NOEVID"
            elif set(claimed) & set(flipped):
                verdict = "caught by %s" % ",".join(flipped)
            else:
                verdict = "NOT CAUGHT (claimed %s)" % ",".join(claimed)
                bad += 1
            print("  %-32s %s" % (name, verdict))
            print("      %s" % rationale)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if bad:
        print("\n%d variant(s) were not caught by a case they name, or could not be applied: the "
              "rationale is wrong, or the case does not discriminate." % bad)
        return 1
    print("\nEvery variant is caught by a case it names. The five real-bug cases are reported and "
          "the three measured false positives are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
