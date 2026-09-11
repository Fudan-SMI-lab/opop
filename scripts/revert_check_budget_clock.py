"""Revert-check for the resume-aware half of `check_wrapup.check_budget_stop`.

A resumed run's `elapsed_hours` counts only from the resume: it is `_elapsed_hours()` off `self.t0`,
set in `Orchestrator.__init__`, and `cmd_resume` builds a NEW orchestrator. Measured live on box 3 --
it reported 7.5 h internally while its first event was 8.75 h old.

The damage is a wall-clock claim, and specifically an ARM-PARITY claim: two arms both reporting "12 h"
having spent different compute is the one thing a paired latency comparison cannot survive, and nothing
in the config diff shows it. So the reader measures the event-timestamp span, which needs no trust in
the payload, and says which figure is which.

Both directions have to hold:

  * detection removed   -> a resumed run reads as an uninterrupted one, and its understated hours go
                           into the paper beside an honest arm's
  * span faked          -> the warning fires with a number that is not the span, which is worse than
                           no number: it looks measured

EACH VARIANT IS PROBED WHERE IT ACTS, on a fixture that reaches its branch.

    python scripts/revert_check_budget_clock.py
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

VARIANTS = {
    "interrupt_never_counted": (
        ('        if e.get("type") == "RUN_INTERRUPTED":\n'
         "            interrupts += 1\n"
         "            continue",
         '        if e.get("type") == "RUN_INTERRUPTED":\n'
         "            continue"),
        "stops noticing the resume. Silent, and it lands exactly where it does the most harm: the "
        "paired report then says 'no resume on either arm' and both arms' hours are compared as if "
        "they were the same clock. Box 3 is a live instance -- 7.5 h reported, 8.75 h spent",
        ["a_resumed_run_is_flagged_because_its_own_elapsed_hours_restarts"], "resumed"),
    "span_from_the_payload_instead": (
        ("    span = (t_last - t_first) / 3600.0 "
         "if (t_first is not None and t_last is not None) else None",
         "    span = stops[0][\"elapsed_hours\"] if stops else None"),
        "reports the run's OWN figure as if it were the measured span, so the warning fires carrying "
        "the very number it exists to correct. A wrong number that looks measured is worse than a "
        "missing one -- the recorded `a-constant-reading-is-a-broken-probe` failure",
        ["the_event_span_is_measured_not_taken_from_the_payload",
         "a_resumed_run_is_flagged_because_its_own_elapsed_hours_restarts"], "understated"),
    "every_run_flagged": (
        ("    if interrupts:", "    if True:"),
        "warns on every run, resumed or not. A warning that always fires carries no information and "
        "will be skipped by the reader on the one run where it matters",
        ["an_uninterrupted_run_is_not_flagged_but_still_reports_its_span"], "clean"),
}


def _ev(kind: str):
    base = {"type": "BASELINE_DONE", "payload": {"baseline": {
        "kind": "eager", "latency_ms": {"mean": 21.5, "median": 21.4, "std": 0.05, "min": 21.2,
                                        "max": 21.6, "n_samples": 100}}}}
    rnd = {"type": "FAMILY_ROUND_RECORDED", "payload": {"family_id": "f", "best_ms": 1.0,
                                                        "round": 1, "conversion": "improved"}}
    intr = {"type": "RUN_INTERRUPTED", "payload": {"reason": "terminated by signal or Ctrl-C"}}
    stop = {"type": "WALL_CLOCK_REACHED", "payload": {"elapsed_hours": 12.0, "budget_hours": 12.0,
                                                      "round": 3, "stopped_before_family": "fam-b"}}
    if kind == "resumed":         # resumed, no stop -- only the interrupt branch shows
        return [base, intr, rnd], 8.75
    if kind == "understated":     # resumed AND stopped: payload says 12 h, events span 20 h
        return [base, intr, stop, rnd], 20.0
    if kind == "clean":           # never resumed -- the must-NOT-fire case
        return [base, rnd], 3.5
    raise AssertionError("unknown probe kind %r" % kind)


def _fixture(kind: str, tmp: Path) -> Path:
    d = tmp / kind
    d.mkdir(parents=True)
    evs, hours = _ev(kind)
    t0 = 1789000000.0
    n = max(len(evs) - 1, 1)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(evs):
            fh.write(json.dumps({"seq": i, "ts": t0 + (hours * 3600.0) * i / n, **e}) + "\n")
    return d


_CODE = ("import sys; sys.path.insert(0, 'scripts'); import check_wrapup\n"
         "from pathlib import Path\n"
         "o = check_wrapup.check_budget_stop(Path(sys.argv[1]))\n"
         "print('interrupts=%s span=%s | %s'\n"
         "      % (o.get('interrupts'), o.get('event_span_hours'), o['verdict'][-105:]))\n")


def _probe(check_dir: Path, kind: str) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="bc-probe-"))
    try:
        d = _fixture(kind, tmp)
        r = subprocess.run(
            [sys.executable, "-B", "-c", _CODE, str(d)], cwd=check_dir, capture_output=True,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
        return (r.stdout.decode("utf-8", "replace").strip()
                or "ERR: " + r.stderr.decode("utf-8", "replace").strip()[-220:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    kinds = ("resumed", "understated", "clean")
    baselines = {k: _probe(ROOT, k) for k in kinds}
    print("BASELINE probe readings (one per branch, so each variant is probed where it acts):")
    for k in kinds:
        print("  %-12s %s" % (k, baselines[k][:150]))
    print()
    bad = 0
    for name, ((old, new), why, claimed, kind) in VARIANTS.items():
        if src.count(old) != 1:
            print("  %-32s SKIPPED -- anchor occurs %d times" % (name, src.count(old)))
            print("      %s" % why)
            bad += 1
            continue
        io.open(TARGET, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        try:
            probe = _probe(ROOT, kind)
            if probe == baselines[kind]:
                print("  %-32s SHAM -- reading identical to the baseline on the %s probe, so it "
                      "patches nothing reachable" % (name, kind))
                print("      %s" % why)
                bad += 1
                continue
            print("  %-32s changes the %s probe:" % (name, kind))
            print("      was: %s" % baselines[kind][:130])
            print("      now: %s" % probe[:130])
            flipped = []
            skipped = []
            for t in claimed:
                r = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", TESTS, "-q", "--no-header",
                     "-p", "no:cacheprovider", "-k", t],
                    cwd=ROOT, capture_output=True,
                    env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"})
                out = r.stdout.decode("utf-8", "replace")
                ran = " no tests ran" not in out
                if not ran:
                    skipped.append(t)
                elif r.returncode != 0:
                    flipped.append(t)
            if flipped:
                print("      CAUGHT by %s" % ", ".join(x[:58] for x in flipped))
            elif skipped:
                print("      UNVERIFIED -- every claimed test SKIPPED here (%s); a skip is not a "
                      "verdict" % ", ".join(x[:40] for x in skipped))
                bad += 1
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
