"""Revert-check for the measured latency floor's `extra_globs` / `--floor-from` widening.

The thing being protected is a TOLERANCE. J2-5 asks whether the two arms' `final_reeval_ms` differ by
more than the noise floor, and the floor this project had been using -- 2.35% -- is `1 - 0.9765`, a
`frac_within_tol` figure: a fraction of ELEMENTS agreeing, borrowed as a timing tolerance. The real
same-kernel re-measurement spread on box 1 is 0.29-4.73%, median 2.91%. So the borrowed number is
BELOW the median of the measurement it stands in for, and a verdict against it alone is stricter than
the data supports.

Every failure here is silent and one-directional -- it makes the comparison stricter, so it can only
manufacture a difference, never hide one. Nothing crashes and no number looks wrong.

EACH VARIANT IS PROBED WHERE IT ACTS. An earlier harness in this project patched an `if` to `False`
and left the `elif`/`else` chain to recompute the same answer; the variant was byte-identical to the
fix and its six tests "passed on the defect", which read as the tests being worthless. Worse, the FIRST
version of this very harness used one fixed probe for all three variants and reported two real
variants as SHAM -- one arm layout cannot reach the CLI parser and the double-count guard at once.
Hence `PROBES`: a fixture and a call per branch.

    python scripts/revert_check_latency_floor_globs.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])
TARGET = ROOT / "scripts" / "check_wrapup.py"
TESTS = "tests/test_check_wrapup.py"

_PARSE_BLOCK = """    floor_globs: list[str] = []
    positional: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--floor-from" and i + 1 < len(argv):
            floor_globs.append(argv[i + 1])
            i += 2
        elif argv[i].startswith("--floor-from="):
            floor_globs.append(argv[i].split("=", 1)[1])
            i += 1
        else:
            positional.append(argv[i])
            i += 1
    if not positional:
        print(__doc__)
        return 2
    argv = positional
"""

VARIANTS = {
    "extra_globs_ignored": (
        ("    for pattern in extra_globs:", "    for pattern in ():"),
        "accepts the parameter and never looks at it -- which is the state the docstring was in "
        "before this work: promised and unimplemented. The floor comes back None on the live layout "
        "and the caller silently falls back to the borrowed 2.35%, below the measured median",
        ["extra_globs_reaches_a_corpus_the_sibling_scan_cannot"], "unfinished_arm"),
    "flag_never_parsed": (
        (_PARSE_BLOCK, "    floor_globs: list[str] = []\n"),
        "leaves `latency_floor_from_runs(extra_globs=...)` correct but never supplies it from the "
        "command line. The layer ABOVE a fix is the one that goes untested; an unparsed flag looks "
        "exactly like a correct run with a thin sample",
        ["the_floor_from_flag_actually_reaches_the_floor_measurement"], "cli"),
    "seen_guard_removed": (
        ('        if d in seen or not (d / "events.jsonl").exists():',
         '        if not (d / "events.jsonl").exists():'),
        "counts a run once per path that reaches it -- direct pass, sibling scan, then the glob. n "
        "inflates while every delta stays correct, so the provenance overstates how much evidence "
        "bounds the tolerance. This is the failure that a widening invites",
        ["extra_globs_does_not_double_count_a_run_the_sibling_scan_already_found"], "double_count"),
}


def _finished(d: Path, tuned: float, reeval: float) -> None:
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text(json.dumps({
        "seq": 0, "ts": 1789000000.0, "type": "RUN_FINISHED",
        "payload": {"summary": {"best": {"candidate_id": "c", "tuned_ms": tuned,
                                         "final_reeval_ms": reeval, "final_reeval_ok": True}}},
    }) + "\n", encoding="utf-8")


def _unfinished(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("", encoding="utf-8")


# One probe per branch. `unfinished_arm` reproduces the LIVE layout -- arms in `runs-v3/`, both
# unfinished, the finished corpus in `runs-l3/` -- which is the only shape where the sibling scan
# finds nothing and the glob is load-bearing. `cli` is the same layout driven through `main`, the only
# way to reach the flag parser. `double_count` puts the arm INSIDE the globbed directory, the only
# shape where a run is reachable by more than one pass.
_CODE_FLOOR = ("import sys; sys.path.insert(0, 'scripts'); import check_wrapup\n"
               "from pathlib import Path\n"
               "f, p = check_wrapup.latency_floor_from_runs(Path(sys.argv[1]),\n"
               "                                           extra_globs=(sys.argv[2],))\n"
               "print('%s | %s' % (f if f is None else round(f, 3), p))\n")
_CODE_CLI = ("import sys, io, contextlib; sys.path.insert(0, 'scripts'); import check_wrapup\n"
             "buf = io.StringIO()\n"
             "with contextlib.redirect_stdout(buf):\n"
             "    rc = check_wrapup.main([sys.argv[1], sys.argv[2], '--floor-from', sys.argv[3]])\n"
             "keep = [l.strip() for l in buf.getvalue().splitlines()\n"
             "        if 'latency floor' in l or 'tolerance used' in l]\n"
             "print('rc=%s | %s' % (rc, ' ;; '.join(keep)))\n")


def _fixture(kind: str, tmp: Path) -> tuple[str, list[str]]:
    """Build the layout this branch needs; return the code to run and its arguments."""
    if kind in ("unfinished_arm", "cli"):
        _unfinished(tmp / "runs-v3" / "arm-a")
        _unfinished(tmp / "runs-v3" / "arm-b")
        _finished(tmp / "runs-l3" / "run-1", 10.0, 10.4)     # 4.0%
        _finished(tmp / "runs-l3" / "run-2", 10.0, 10.1)     # 1.0%
        pattern = str(tmp / "runs-l3" / "run-*")
        if kind == "cli":
            return _CODE_CLI, [str(tmp / "runs-v3" / "arm-a"),
                               str(tmp / "runs-v3" / "arm-b"), pattern]
        return _CODE_FLOOR, [str(tmp / "runs-v3" / "arm-a"), pattern]
    if kind == "double_count":
        _finished(tmp / "runs" / "run-1", 10.0, 10.4)
        _finished(tmp / "runs" / "run-2", 10.0, 10.1)
        return _CODE_FLOOR, [str(tmp / "runs" / "run-1"), str(tmp / "runs" / "run-*")]
    raise AssertionError("unknown probe kind %r" % kind)


def _probe(check_dir: Path, kind: str) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="floor-probe-"))
    try:
        code, args = _fixture(kind, tmp)
        r = subprocess.run(
            [sys.executable, "-B", "-c", code, *args], cwd=check_dir, capture_output=True,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
        out = r.stdout.decode("utf-8", "replace").strip()
        return out or ("ERR: " + r.stderr.decode("utf-8", "replace").strip()[-220:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    baselines = {k: _probe(ROOT, k) for k in ("unfinished_arm", "cli", "double_count")}
    print("BASELINE probe readings (one per branch, so each variant is probed where it acts):")
    for k, v in baselines.items():
        print("  %-14s %s" % (k, v[:150]))
    print()
    bad = 0
    for name, ((old, new), why, claimed, kind) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-22s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            probe = _probe(ROOT, kind)
            if probe == baselines[kind]:
                print("  %-22s SHAM -- reading identical to the baseline on the %s probe, so it "
                      "patches nothing reachable" % (name, kind))
                print("      %s" % why)
                bad += 1
                continue
            print("  %-22s changes the %s probe:" % (name, kind))
            print("      was: %s" % baselines[kind][:130])
            print("      now: %s" % probe[:130])
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
                print("      CAUGHT by %s" % ", ".join(x[:60] for x in flipped))
            else:
                print("      *** NOT CAUGHT *** (claimed %s)" % claimed)
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
