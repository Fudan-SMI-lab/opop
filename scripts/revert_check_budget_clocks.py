"""The budget-clock reader must not conflate the two clocks.

`_elapsed_hours()` measures from `self.t0 = time.monotonic()`, set in `Orchestrator.__init__`, and
`cmd_resume` builds a NEW Orchestrator -- so a resumed run's budget clock restarts at zero while its
wall age keeps counting. Both numbers are true and they answer different questions:

  * reporting only WALL age understates how much budget a resumed run has left, which predicts it
    will stop hours before it does;
  * reporting only BUDGET age hides that a resumed arm got more total wall clock than its partner,
    which is a cross-arm confound -- the trials already on disk are not re-run, so the extra time
    buys extra SEARCH.

Every scenario below is checked in both directions, because a reader that always reports the two as
equal passes any single-direction test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "check_budget_clocks.py"

_H = 3600.0


def _write(d: Path, events: list[dict], budget: float | None = 12.0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps({"seq": i, **e}) + "\n")
    if budget is not None:
        # The real nesting, verified on box 1: manifest.json -> config -> budgets.
        (d / "manifest.json").write_text(json.dumps({
            "created": "2026-09-11T05:30:20Z",
            "task": {"task_id": "level3:43"},
            "config": {"budgets": {"wall_clock_hours": budget, "trials_per_space": 40}},
        }), encoding="utf-8")
    return d


def _plain(hours: float) -> list[dict]:
    """A run that was never interrupted: both clocks must agree."""
    return [{"ts": 0.0, "type": "RUN_CREATED", "payload": {}},
            {"ts": hours * _H, "type": "TRIAL_DONE", "payload": {"trial": {"status": "complete"}}}]


def _resumed(before: float, gap: float, after: float) -> list[dict]:
    """`before` hours, then RUN_INTERRUPTED, a `gap`, then `after` hours in the new process."""
    t = 0.0
    evs = [{"ts": t, "type": "RUN_CREATED", "payload": {}}]
    t += before * _H
    evs.append({"ts": t, "type": "RUN_INTERRUPTED",
                "payload": {"reason": "terminated by signal or Ctrl-C"}})
    t += gap * _H
    evs.append({"ts": t, "type": "SPACE_PRESCREENED", "payload": {"configs_probed": 40}})
    t += after * _H
    evs.append({"ts": t, "type": "TRIAL_DONE", "payload": {"trial": {"status": "complete"}}})
    return evs


def _run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-B", str(CHK), *args],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return proc.returncode, proc.stdout + proc.stderr


def _nums(out: str, label: str) -> tuple[float, float, float] | None:
    """(wall_h, budget_h, left_h) off the row for `label`."""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == label:
            try:
                return float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                return None
    return None


# (label, builder -> dict of arm-name: (events, budget), assertion(out) -> str|None)
def _sc_plain(tmp: Path):
    d = _write(tmp / "p", _plain(3.0))
    rc, out = _run(["ARM", str(d)])
    n = _nums(out, "ARM")
    if n is None:
        return "no ARM row parsed:\n%s" % out
    wall, bud, left = n
    if abs(wall - 3.0) > 0.02 or abs(bud - 3.0) > 0.02:
        return "an uninterrupted run must have equal clocks, got wall=%.2f budget=%.2f" % (wall, bud)
    if abs(left - 9.0) > 0.02:
        return "left_h wrong: expected 12 - 3 = 9, got %.2f" % left
    if "RESUMED" in out:
        return "an uninterrupted run was reported as resumed"
    return None


def _sc_resumed(tmp: Path):
    # 1.3 h before the interrupt, 0.02 h gap, 1.5 h after: wall 2.82, budget 1.5.
    d = _write(tmp / "r", _resumed(1.3, 0.02, 1.5))
    rc, out = _run(["ARM", str(d)])
    n = _nums(out, "ARM")
    if n is None:
        return "no ARM row parsed:\n%s" % out
    wall, bud, left = n
    if abs(wall - 2.82) > 0.03:
        return "wall age wrong: expected 2.82, got %.2f" % wall
    if abs(bud - 1.5) > 0.03:
        return "budget age wrong: it must restart at the resume, expected 1.50, got %.2f" % bud
    if abs(left - 10.5) > 0.03:
        return "left_h must use the BUDGET clock: expected 10.50, got %.2f" % left
    if "RESUMED" not in out:
        return "the resume was not flagged, so the extra wall clock is invisible"
    if "1.32" not in out:
        return "the amount of extra wall clock (1.32 h) is not stated: '%s'" % out.strip()[-200:]
    return None


def _sc_cross_arm(tmp: Path):
    a = _write(tmp / "a", _plain(3.0))
    b = _write(tmp / "b", _resumed(1.3, 0.02, 1.5))
    rc, out = _run(["CTL", str(a), "TRT", str(b)])
    if "asymmetry" not in out.lower():
        return "a pair with one resumed arm did not warn about the budget asymmetry"
    if "TRT" not in out.split("asymmetry")[1][:200]:
        return "the warning does not name which arm was resumed"
    return None


def _sc_no_manifest(tmp: Path):
    d = _write(tmp / "n", _plain(3.0), budget=None)
    rc, out = _run(["ARM", str(d)])
    n = _nums(out, "ARM")
    if n is None:
        return "no ARM row parsed with the manifest absent:\n%s" % out
    wall, bud, left = n
    if abs(wall - 3.0) > 0.02:
        return "the clocks must still be read without a manifest, got wall=%.2f" % wall
    if left == left:  # not nan
        return ("with no manifest there is no configured budget, so left_h must be nan rather "
                "than a number, got %.2f" % left)
    return None


SCENARIOS = [
    ("an uninterrupted run has one clock", _sc_plain,
     "both clocks must agree and left_h must come off the configured budget"),
    ("a resumed run has two clocks", _sc_resumed,
     "the budget clock restarts (Orchestrator.__init__ resets t0 and cmd_resume builds a new one) "
     "while the wall age keeps counting; both must be reported and the difference stated"),
    ("a pair with one resumed arm warns", _sc_cross_arm,
     "the resumed arm got more total wall clock than its partner, and the extra time buys extra "
     "SEARCH because the trials already on disk are not re-run"),
    ("a missing manifest yields nan, not a number", _sc_no_manifest,
     "no configured budget means the remaining budget is unknown; printing a number would invent "
     "one, and 0 or 12 are both wrong in a way that looks fine"),
]

VARIANTS: list[tuple[str, str, str, list[str], str]] = [
    (
        "the budget clock never restarts: a resume read as one long run",
        "    budget_t0 = first\n"
        "    if resume_at is not None:",
        "    budget_t0 = first\n"
        "    if False:",
        ["a resumed run has two clocks"],
        "reports budget_h == wall_h for a resumed run, so box 3 reads 2.81 h used of 12 instead of "
        "1.53 h. That UNDERSTATES its remaining budget by 1.28 h and predicts it stops hours before "
        "it will -- and a monitoring decision ('it will not reach Loop C') would be made on the "
        "wrong number",
    ),
    (
        "the wall age replaced by the budget age",
        '    return {"wall_h": (last - first) / 3600.0,',
        '    return {"wall_h": (last - budget_t0) / 3600.0,',
        ["a resumed run has two clocks"],
        "both columns show the budget clock, so the extra wall clock a resumed arm received becomes "
        "invisible. That is the cross-arm confound: the resumed arm searched for longer in total, "
        "and no number on the report says so. (Only the single-arm scenario is named: the pair "
        "scenario asserts the WARNING text, which still fires because `interrupts` is unaffected -- "
        "measured, not assumed.)",
    ),
    (
        "the resume flag dropped",
        '        if r["interrupts"]:',
        "        if False:",
        ["a resumed run has two clocks"],
        "the two clocks are still printed but nothing says why they differ. A reader comparing "
        "2.81 against 1.53 has to know that `cmd_resume` rebuilds the Orchestrator to interpret it, "
        "and the whole point of the row is that this is not obvious. (The pair scenario still "
        "passes: the cross-arm warning is emitted by a separate branch keyed on `interrupts`, so "
        "per-row silence and pair-level silence are independent failures and each needs its own "
        "variant -- which is why the next one exists.)",
    ),
    (
        "the cross-arm asymmetry warning dropped",
        "    if resumed and len(rows) > 1:",
        "    if False:",
        ["a pair with one resumed arm warns"],
        "each arm's row is right and the comparison between them is unguarded. J2-5 compares the "
        "arms' final latency, so 'the same wall clock' has to be true or stated false -- the same "
        "reason `check_arm_search_parity.py` exists",
    ),
    (
        "the budget defaults to a number when the manifest is absent",
        "        r = read(p)\n"
        '        r["configured_budget_h"] = b',
        "        r = read(p)\n"
        '        r["configured_budget_h"] = b if b is not None else 12.0',
        ["a missing manifest yields nan, not a number"],
        "invents a budget. 12.0 is the value these runs happen to use, so the number looks right "
        "and is unfalsifiable from the output -- the most durable kind of wrong number, and the "
        "same shape as the 2.35% correctness figure that was serving as a latency tolerance",
    ),
    (
        "the first event after the interrupt taken as the interrupt itself",
        "        later = [e[\"ts\"] for e in evs if e[\"ts\"] > resume_at]\n"
        "        budget_t0 = min(later) if later else resume_at",
        "        budget_t0 = resume_at",
        [],
        "counts the DOWNTIME between the interrupt and the restart as budget spent. Measured: box "
        "3's gap is ~0.02 h, so on this run the two agree to within the print precision and no "
        "scenario can flip -- NOEVID by construction. It would matter for a run resumed hours "
        "later, which is why the reader keeps the distinction rather than simplifying to the "
        "interrupt timestamp",
    ),
]


def main() -> int:
    original = CHK.read_text(encoding="utf-8")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("baseline:")
        base: dict[str, str | None] = {}
        for label, fn, _why in SCENARIOS:
            err = fn(tmp / ("base_" + label.replace(" ", "_")[:30]))
            base[label] = err
            print("  %-44s %s" % (label, "ok" if err is None else "**FAILS**: " + err))
            if err is not None:
                ok = False
        if not ok:
            print("\nBASELINE IS NOT CLEAN -- nothing below means anything")
            return 2
        print()

        try:
            for label, old, new, must_break, why in VARIANTS:
                if original.count(old) != 1:
                    print("**SKIPPED** %s: anchor occurs %d times, not once"
                          % (label, original.count(old)))
                    ok = False
                    continue
                patched = original.replace(old, new)
                try:
                    compile(patched, str(CHK), "exec")
                except SyntaxError as exc:
                    print("**SKIPPED** %s: patched file does not parse (%s)" % (label, exc))
                    ok = False
                    continue
                CHK.write_text(patched, encoding="utf-8")
                try:
                    got = {}
                    for slabel, fn, _w in SCENARIOS:
                        got[slabel] = fn(tmp / (label.replace(" ", "_")[:24] + "_"
                                                + slabel.replace(" ", "_")[:24]))
                finally:
                    CHK.write_text(original, encoding="utf-8")

                survived = [s for s in must_break if got.get(s) is None]
                if not must_break:
                    print("NOEVID  %s" % label)
                    print("        names no scenario: measured that no verdict changes")
                elif survived:
                    ok = False
                    print("**FAIL** %s" % label)
                    print("        these scenarios still PASSED on the broken reader, so they are")
                    print("        not evidence for it: %s" % ", ".join(survived))
                else:
                    print("ok      %s" % label)
                    print("        %d/%d scenarios broke as required" % (
                        len(must_break), len(must_break)))
                print("        wrong version: %s" % why)
                print()
        finally:
            CHK.write_text(original, encoding="utf-8")

        for label, fn, _why in SCENARIOS:
            if fn(tmp / ("after_" + label.replace(" ", "_")[:30])) is not None:
                print("!! RESTORE DID NOT COME BACK CLEAN -- check git status")
                return 2
    print("restored, baseline clean again")
    if not ok:
        print("\nVERDICT: at least one scenario is not evidence -- see above")
        return 1
    print("\nVERDICT: every scenario breaks on the reader it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
