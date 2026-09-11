"""The parity check must not report OK on arms that did unequal search.

The reader here has the same dangerous failure mode as the rest: a wrong reading produces a
confident verdict rather than an error. Two directions matter and they are NOT symmetric:

  * a FALSE "PARITY OK" certifies a confounded comparison, which is how a paper reports a switch's
    effect that was actually extra search;
  * a FALSE "PARITY NOT OK" fires on every mid-run invocation (the space in flight is always short)
    and on the one 1800 s timeout that dominated one arm's gap sum, and a check that always
    complains is a check whose output stops being read.

Both were live bugs in the first draft: the tail test came from finding that a 2.63x mean rate ratio
collapsed to 23.8 s vs 21.3 s at the median, with a single timeout holding 60% of the total.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "check_arm_search_parity.py"


def _write(d: Path, events: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps({"seq": i, **e}) + "\n")
    return d


def _trial(ts: float, space: str, fk: str | None = None, dtype: str = "fp16") -> dict:
    return {"ts": ts, "type": "TRIAL_DONE", "payload": {"trial": {
        "space_id": space, "status": "complete", "failure_kind": fk,
        "params": {"values": {"COMPUTE_DTYPE": dtype, "DOT_MODE": "plain"}},
        "latency_ms": {"median": 3.0}}}}


def _arm(n_spaces: int, per_space: int, step: float, tail: float = 0.0) -> list[dict]:
    """n_spaces full spaces of `per_space` trials each, one trial every `step` seconds, plus an
    optional single pathological trial of length `tail`."""
    evs: list[dict] = [{"ts": 0.0, "type": "RUN_CREATED", "payload": {}}]
    t = 0.0
    for s in range(n_spaces):
        for _ in range(per_space):
            t += step
            evs.append(_trial(t, "sp-%d" % s, None))
    if tail:
        t += tail
        evs.append(_trial(t, "sp-0", "timeout"))
    return evs


# Each scenario: (label, control events, treatment events, what the verdict must be, why)
SCENARIOS = [
    (
        "equal arms",
        _arm(2, 40, 30.0), _arm(2, 40, 30.0), True,
        "the null case: identical rate and identical per-space budget must certify as attributable",
    ),
    (
        "one arm slower in the BODY (no tail)",
        _arm(2, 40, 60.0), _arm(2, 40, 30.0), False,
        "every trial twice as slow with no pathological config: the fast arm fits twice the search "
        "into the same wall clock, so a latency difference has two explanations. This is the "
        "scenario the whole check exists for",
    ),
    (
        "one arm slowed only by a single timeout",
        _arm(2, 40, 30.0, tail=1800.0), _arm(2, 40, 30.0), True,
        "MEASURED on the live arms: a 1.76x rate ratio whose slower side had 68% of its gap sum in "
        "the top 10%, from one 1800 s timeout. A tail does not compound over 12 h, so calling this "
        "a confound would report a defect on a healthy pair",
    ),
    (
        "unequal per-space budget",
        _arm(2, 40, 30.0), _arm(2, 20, 30.0), False,
        "the same rate but half the trials per space: the budget itself is applied unequally, which "
        "is worse than a rate difference and is invisible to any rate comparison",
    ),
    (
        "treatment mid-space (short final space)",
        _arm(2, 40, 30.0), _arm(2, 40, 30.0) + [_trial(3000.0, "sp-9")], True,
        "every mid-run invocation looks like this -- the space in flight is short by definition. "
        "Firing here makes the check useless while the runs are alive, which is exactly when it is "
        "consulted",
    ),
    (
        "same rate, staggered start (the real pair launched 4 min apart)",
        # 40 trials vs 80, both at one per 30 s: identical trials/h, a 2.0x raw COUNT ratio. The
        # stagger has to be big enough that the count ratio clears the 15% tolerance while the rate
        # ratio stays at 1.0 -- a one-trial stagger (81 vs 80) is 1.25% and clears nothing, which is
        # why the first version of this scenario could not discriminate anything.
        _arm(1, 40, 30.0), _arm(2, 40, 30.0), True,
        "identical PER-HOUR rate but 2x the absolute trials, because one arm has been observed for "
        "twice as long. The live pair launched four minutes apart and is read mid-run, so every "
        "comparison has this shape -- a checker comparing raw counts fails all of them",
    ),
]

# Variants of the checker itself, each with the scenarios whose verdict it must change.
VARIANTS: list[tuple[str, str, str, list[str], str]] = [
    (
        "the tail exemption removed: a timeout counted as a rate difference",
        "            if body is not None and body <= _RATE_TOL:",
        "            if False:",
        ["one arm slowed only by a single timeout"],
        "reports PARITY NOT OK on the live pair, whose 1.76x ratio is one 1800 s timeout holding "
        "60% of one arm's gap sum. That is a false confound report on a healthy pair -- and the "
        "response to a false confound is to discard a valid result",
    ),
    (
        "the tail exemption always taken: a genuinely slower box excused",
        "            if body is not None and body <= _RATE_TOL:",
        "            if True:",
        ["one arm slower in the BODY (no tail)"],
        "the dangerous direction. Certifies a 2x rate difference as attributable, so the paper "
        "reports the switch's effect when half the difference is that one arm searched twice as "
        "much. A tail-share test that never says no is not a test",
    ),
    (
        "the rate comparison dropped entirely",
        "        if ratio > _RATE_TOL:",
        "        if False:",
        ["one arm slower in the BODY (no tail)"],
        "every pair certifies. This is the state before this script existed: config equality was "
        "verified field-by-field (`audit_arm_comparability.py`) and read as settling comparability, "
        "but equal configs with unequal rates still buy unequal search",
    ),
    (
        "the per-space budget check dropped",
        "    if ma is not None and mb is not None and ma != mb:",
        "    if False:",
        ["unequal per-space budget"],
        "half the trials per space passes as parity because the RATE is identical. The two limits "
        "are independent: `trials_per_space` and `wall_clock_hours`, and a rate comparison cannot "
        "see the first",
    ),
    (
        "modal budget replaced by the min: the in-flight space compared as if finished",
        "    ma, mb = _mode(ca), _mode(cb)",
        "    ma, mb = (min(ca) if ca else None), (min(cb) if cb else None)",
        ["treatment mid-space (short final space)"],
        "compares the SHORTEST space, which mid-run is always the one in flight. Fires on every "
        "live invocation, and a check that always complains stops being read -- the same reason "
        "`check_wrapup.py` refuses to call 0-of-0 rounds a defect",
    ),
    (
        "modal budget replaced by the max: an over-budget space read as unequal",
        "    ma, mb = _mode(ca), _mode(cb)",
        "    ma, mb = (max(ca) if ca else None), (max(cb) if cb else None)",
        ["one arm slowed only by a single timeout"],
        "a space can EXCEED the nominal budget -- a timeout still lands a TRIAL_DONE -- so the max "
        "reports 41 vs 40 and calls a healthy pair unequal. Measured: the live control arm has 41 "
        "trials in one space because of its one timeout",
    ),
    (
        # MEASURED, not reasoned: patching the ratio to raw counts and running the staggered
        # scenario prints "rate 120.0 vs 120.0 trials/h (2.00x)" and then PARITY OK, because the
        # median-gap guard immediately exempts it (30.0 vs 30.0 s). The rate metric is not
        # load-bearing on its own -- the median gap is what decides -- so no single mutation of the
        # ratio can flip a verdict, and a variant claiming otherwise would be a false ok. Kept as a
        # NOEVID entry rather than deleted, because "this line looked load-bearing and is not" is
        # the finding.
        "rate measured on trials rather than trials per hour",
        '        ratio = max(ra, rb) / min(ra, rb)',
        "        ratio = max(a['trials'], b['trials']) / min(a['trials'], b['trials'])",
        [],
        "counts absolute trials, so a staggered pair reads as unequal -- but the median-gap guard "
        "downstream exempts it anyway (verified: the patched checker prints 2.00x and still says "
        "PARITY OK). The verdict is decided by the median gap, not by this ratio; the ratio only "
        "chooses the wording. NOEVID by construction, and that is the point",
    ),
]


def _slug(label: str) -> str:
    """A filesystem-safe directory name. Windows rejects ':' in a path, and several variant labels
    contain one -- which crashed the harness after the baseline had already passed."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]


def _scenario_dirs(base: Path, ctl: list[dict], trt: list[dict]) -> tuple[Path, Path]:
    return (_write(base / "ctl", ctl), _write(base / "trt", trt))


def run_scenarios(tmp: Path) -> dict[str, bool]:
    """Returns {scenario label: verdict was OK}."""
    out: dict[str, bool] = {}
    for label, ctl, trt, _expect, _why in SCENARIOS:
        d = tmp / _slug(label)
        a, b = _scenario_dirs(d, ctl, trt)
        proc = subprocess.run([sys.executable, "-B", str(CHK), str(a), str(b)],
                              capture_output=True, text=True, cwd=ROOT,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        text = proc.stdout + proc.stderr
        if proc.returncode not in (0, 1):
            out[label] = None  # crashed: neither verdict
            print("        (scenario %r crashed: %s)" % (label, text.strip()[-200:]))
            continue
        out[label] = proc.returncode == 0
    return out


def main() -> int:
    original = CHK.read_text(encoding="utf-8")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = run_scenarios(tmp / "base")
        print("baseline verdicts:")
        for label, _c, _t, expect, _why in SCENARIOS:
            got = base.get(label)
            mark = "ok" if got is expect else "**WRONG**"
            print("  %-42s expected %-5s got %-5s %s" % (
                label, "OK" if expect else "NOT-OK",
                {True: "OK", False: "NOT-OK", None: "CRASH"}[got], mark))
            if got is not expect:
                ok = False
        if not ok:
            print("\nBASELINE IS WRONG -- nothing below means anything")
            return 2
        print()

        try:
            for label, old, new, must_flip, why in VARIANTS:
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
                    got = run_scenarios(tmp / _slug(label))
                finally:
                    CHK.write_text(original, encoding="utf-8")

                unflipped = [s for s in must_flip if got.get(s) is base.get(s)]
                if not must_flip:
                    # A variant that names no scenario is not evidence and must never print "ok" --
                    # the recorded rule is that a variant naming no test prints NOEVID.
                    print("NOEVID  %s" % label)
                    print("        names no scenario: verified by measurement that no verdict flips")
                elif unflipped:
                    ok = False
                    print("**FAIL** %s" % label)
                    print("        these scenarios gave the SAME verdict on the broken checker, so")
                    print("        they are not evidence for it: %s" % ", ".join(unflipped))
                else:
                    print("ok      %s" % label)
                    print("        %d/%d scenarios flipped as required" % (
                        len(must_flip), len(must_flip)))
                print("        wrong version: %s" % why)
                print()
        finally:
            CHK.write_text(original, encoding="utf-8")

    after = run_scenarios(Path(tempfile.mkdtemp()))
    if any(after.get(l) is not e for l, _c, _t, e, _w in SCENARIOS):
        print("!! RESTORE DID NOT COME BACK CLEAN -- check git status")
        return 2
    print("restored, baseline verdicts reproduce")
    if not ok:
        print("\nVERDICT: at least one scenario is not evidence -- see **FAIL** above")
        return 1
    print("\nVERDICT: every scenario flips on the checker it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
