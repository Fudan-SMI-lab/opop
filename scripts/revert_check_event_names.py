"""Revert-check for scripts/check_event_names.py.

Every variant below is a plausible simplification of the checker, and each must be caught by a
scenario NAMED in its rationale. The rule this file exists to enforce: a variant that no scenario
flips is NOT evidence the checker works -- it prints NOEVID rather than "ok", because an "ok" on a
broken variant is worse than no harness at all.

The checker is run against SYNTHETIC trees whose answer is known by construction. Running it against
the live repo can only ever confirm that it prints what it prints -- which is the same trap the
checker itself was written for: a fixture built from the reader's expectations proves nothing.

    python scripts/revert_check_event_names.py
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
CHK = ROOT / "scripts" / "check_event_names.py"

# 16 emitted names: one above the checker's own `< 15` positive-control floor, so the clean
# scenarios exercise the real path and scenario `probe_broken` can drop below it deliberately.
_EMITTED = [
    "RUN_CREATED", "RUN_FINISHED", "TRIAL_DONE", "BASELINE_DONE", "STEP_DONE",
    "CANDIDATE_REGISTERED", "SPACE_PUBLISHED", "SPACE_PRESCREENED", "TUNING_DONE", "STATS_DONE",
    "REWRITE_PRODUCED", "CONVERGENCE_DECIDED", "AGENT_CALL_FAILED", "DIMENSION_STATE",
    "BOTTLENECK_CLASSIFIED", "WALL_CLOCK_REACHED",
]


def _src(names: list[str]) -> str:
    body = "\n".join('    store.append("%s", {})' % n for n in names)
    return "def emit(store):\n" + body + "\n"


def _tree(reader: str, *, emitted: list[str] | None = None,
          reader_name: str = "reader.py") -> Path:
    d = Path(tempfile.mkdtemp(prefix="cen-"))
    (d / "src").mkdir()
    (d / "scripts").mkdir()
    (d / "tests").mkdir()
    (d / "src" / "emitter.py").write_text(_src(_EMITTED if emitted is None else emitted),
                                          encoding="utf-8")
    (d / "scripts" / reader_name).write_text(reader, encoding="utf-8")
    return d


# --- scenarios: each returns (tree, expected exit code, a predicate on stdout) -----------------

def sc_clean() -> tuple[Path, int, str]:
    """A reader keyed only on names src/ emits. Nothing to report."""
    return _tree(
        'def go(e):\n'
        '    t = e["type"]\n'
        '    if t == "TRIAL_DONE":\n        pass\n'
        '    elif t in ("REWRITE_PRODUCED", "STEP_DONE"):\n        pass\n'
    ), 0, "EVERY event name used by a reader is emitted"


def sc_sole_key() -> tuple[Path, int, str]:
    """A dead name that is the ONLY key for its branch: the branch can never run."""
    return _tree(
        'def go(e):\n'
        '    t = e["type"]\n'
        '    if t == "FINAL_REEVAL_DONE":\n        return "never"\n'
    ), 1, "SOLE key"


def sc_alternation() -> tuple[Path, int, str]:
    """A dead name ALTERNATED with a real one still counts the real events: dead weight only."""
    return _tree(
        'def go(e):\n'
        '    t = e["type"]\n'
        '    if t in ("REWRITE_PRODUCED", "REWRITE_ROUND_DONE"):\n        pass\n'
    ), 0, "Dead weight, not a defect"


def sc_split_branches() -> tuple[Path, int, str]:
    """Real name and dead name in the SAME FILE but on DIFFERENT lines, each its own branch.

    A whole-file notion of "guarded" excuses this; it is a real defect -- the second branch is
    unreachable regardless of what the first one matches.
    """
    return _tree(
        'def go(e):\n'
        '    t = e["type"]\n'
        '    if t == "TRIAL_DONE":\n        pass\n'
        '    elif t == "FINAL_REEVAL_DONE":\n        return "never"\n'
    ), 1, "SOLE key"


def sc_numeric_tuple() -> tuple[Path, int, str]:
    """`for t in (600.0, 1500.0)` matches the membership pattern and is not an event name."""
    return _tree(
        'def go(e):\n'
        '    for t in (600.0, 1500.0, 1800.0):\n        print(t)\n'
        '    if e["type"] == "TRIAL_DONE":\n        pass\n'
    ), 0, "EVERY event name used by a reader is emitted"


def sc_revert_check_exempt() -> tuple[Path, int, str]:
    """A revert_check_* file names a wrong event ON PURPOSE -- that is the bug it injects."""
    return _tree(
        'def variant(e):\n'
        '    if e["type"] == "FINAL_REEVAL_DONE":\n        return "injected bug"\n',
        reader_name="revert_check_thing.py",
    ), 0, "EVERY event name used by a reader is emitted"


def sc_probe_broken() -> tuple[Path, int, str]:
    """Too few emitted names means the EXTRACTION broke; every "missing" would be an artefact."""
    return _tree(
        'def go(e):\n'
        '    if e["type"] == "FINAL_REEVAL_DONE":\n        pass\n',
        emitted=["RUN_CREATED", "TRIAL_DONE"],
    ), 2, "PROBE BROKEN"


def sc_knob_not_event() -> tuple[Path, int, str]:
    """An upper-case literal in a NON-event position (a param knob) must not be reported."""
    return _tree(
        'def go(e, params):\n'
        '    x = params["BLOCK_M"] * params["QKV_BLOCK_N"]\n'
        '    labels = ("COMPUTE_DTYPE", "DOT_MODE")\n'
        '    if e["type"] == "TRIAL_DONE":\n        return x, labels\n'
    ), 0, "EVERY event name used by a reader is emitted"


SCENARIOS = {
    "clean": sc_clean,
    "sole_key": sc_sole_key,
    "alternation": sc_alternation,
    "split_branches": sc_split_branches,
    "numeric_tuple": sc_numeric_tuple,
    "revert_check_exempt": sc_revert_check_exempt,
    "probe_broken": sc_probe_broken,
    "knob_not_event": sc_knob_not_event,
}


# --- variants ----------------------------------------------------------------------------------

def _sub(pat: str, repl: str, *, count: int = 1) -> "callable":
    def apply(text: str) -> str:
        new, n = re.subn(pat, repl, text, count=count, flags=re.M)
        if n != count:
            raise SystemExit("variant pattern did not apply (%d != %d): %.60r" % (n, count, pat))
        return new
    return apply


VARIANTS = {
    "numeric_tuple_kept": (
        _sub(r'    return \[hit\] if _NAME_RE\.match\(hit\) else \[\]',
             '    return [hit]'),
        "drops the guard that a bare capture must LOOK like an event name, so a numeric tuple body "
        "is reported as a missing event",
        ["numeric_tuple"],
    ),
    "guarded_always_true": (
        _sub(r'^def _is_guarded\(path: Path, name: str, emitted: set\[str\]\) -> bool:$',
             'def _is_guarded(path: Path, name: str, emitted: set[str]) -> bool:\n    return True'),
        "treats every dead name as harmless dead weight, so a sole-key unreachable branch is "
        "reported as fine and the script exits 0",
        ["sole_key", "split_branches"],
    ),
    "guarded_always_false": (
        _sub(r'^def _is_guarded\(path: Path, name: str, emitted: set\[str\]\) -> bool:$',
             'def _is_guarded(path: Path, name: str, emitted: set[str]) -> bool:\n    return False'),
        "calls every dead name a defect, so a harmless alternation fails the check and the "
        "distinction the script exists to draw is gone",
        ["alternation"],
    ),
    "guarded_whole_file": (
        _sub(r"^        if '\"%s\"' % name not in line:\n            continue$",
             "        if True:\n            continue"),
        "asks whether the FILE mentions a real event rather than the LINE, so a dead sole-key "
        "branch sitting beside an unrelated good branch is excused",
        ["split_branches", "sole_key"],
    ),
    "no_revert_exempt": (
        _sub(r'^        real = \{p for p in missing\[name\] if not p\.name\.startswith\("revert_check_"\)\}$',
             '        real = set(missing[name])'),
        "reports the wrong event a revert-check injects on purpose, which trains the reader to "
        "ignore this script's output",
        ["revert_check_exempt"],
    ),
    "no_positive_control": (
        _sub(r'^    if len\(emitted\) < 15:$', '    if False:'),
        "removes the floor on how many emitted names a working extraction must find, so a broken "
        "extraction reports every reader as defective instead of reporting itself",
        ["probe_broken"],
    ),
    "hard_exits_zero": (
        _sub(r'^    return 1\n\n\nif __name__', '    return 0\n\n\nif __name__'),
        "still PRINTS the defect but exits 0, so any caller gating on the exit code (a pre-commit "
        "hook, CI) sees a pass",
        ["sole_key", "split_branches"],
    ),
    "emit_only_append": (
        _sub(r'^        names \|= set\(_TYPE_CMP_RE\.findall\(text\)\)$', '        pass'),
        "stops collecting names that src/ compares against rather than appends. NO SCENARIO "
        "COVERS THIS: every synthetic emitter here writes store.append, so nothing flips. On the "
        "real repo both regexes find the same 57 names, so it is untested there too -- it would "
        "only matter for an event src/ reads but never writes, which is a different check",
        [],
    ),
    "drop_not_events": (
        _sub(r'^_NOT_EVENTS = \{$', '_NOT_EVENTS = set()\n_UNUSED_NOT_EVENTS = {'),
        "removes the non-event exemption list. NO SCENARIO FLIPS: measured on the real repo, 0 of "
        "the 36 exemptions currently fire because the patterns were narrowed to event positions, "
        "and knob_not_event passes without the list too. The list is insurance against a future "
        "reader, not load-bearing today -- which is why the checker now prints how many "
        "exclusions actually fired",
        [],
    ),
}


def _run(tree: Path, script: Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-B", str(script), "--root", str(tree)],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return proc.returncode, proc.stdout + proc.stderr


def _check(script: Path) -> dict[str, bool]:
    """Which scenarios PASS with this script? Trees are rebuilt per script for isolation."""
    out: dict[str, bool] = {}
    for name, make in SCENARIOS.items():
        tree, want_rc, want_txt = make()
        try:
            rc, txt = _run(tree, script)
            out[name] = (rc == want_rc) and (want_txt in txt)
        finally:
            shutil.rmtree(tree, ignore_errors=True)
    return out


def main() -> int:
    base = _check(CHK)
    print("BASELINE (scripts/check_event_names.py):")
    for name, ok in base.items():
        print("  %-22s %s" % (name, "pass" if ok else "FAIL"))
    if not all(base.values()):
        print("\nBASELINE IS BROKEN -- the current checker fails a scenario it is supposed to "
              "satisfy. Fix that before reading any variant result: on a broken baseline a variant "
              "'caught' means nothing.")
        return 1

    src = CHK.read_text(encoding="utf-8")
    bad = 0
    print("\nVARIANTS:")
    for name, (apply, rationale, claimed) in VARIANTS.items():
        d = Path(tempfile.mkdtemp(prefix="cen-var-"))
        try:
            v = d / "variant.py"
            v.write_text(apply(src), encoding="utf-8")
            got = _check(v)
            flipped = sorted(k for k, ok in got.items() if base[k] and not ok)
            if not claimed:
                # A variant that names no test must say so out loud. The recorded rule: variants
                # that name no tests print NOEVID, never "ok" -- an "ok" here would read as
                # "checked and fine" when nothing was checked.
                verdict = ("NOEVID" if not flipped
                           else "UNCLAIMED-CATCH by %s" % ",".join(flipped))
            elif set(claimed) & set(flipped):
                verdict = "caught by %s" % ",".join(flipped)
            else:
                verdict = "NOT CAUGHT (claimed %s)" % ",".join(claimed)
                bad += 1
            print("  %-22s %s" % (name, verdict))
            print("      %s" % rationale)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    if bad:
        print("\n%d variant(s) claimed a scenario that does not actually flip: the rationale is "
              "wrong, or the scenario does not discriminate." % bad)
        return 1
    print("\nEvery claiming variant is caught by a scenario it names. Two variants name none and "
          "print NOEVID, with the measurement that led to that in their rationale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
