"""The Loop C projector must not mis-count candidates or extrapolate a one-off tail.

Three failure modes, all found on live data rather than imagined:

  * `BOTTLENECK_CLASSIFIED` fires TWICE per candidate whose space gets expanded (once after the
    initial tuning, once after the re-tune). Counting it as "candidates done" overcounts -- I
    reported box 2 as "3 of 4 classified" from that count and it was 2 of 4. It also halves the
    apparent per-candidate cost, because the first classification lands before the re-tune.
  * `CANDIDATE_REGISTERED` fires for all four seeds at t~0. Keying "work began" on it reports every
    unstarted candidate as in flight since the beginning AND subtracts that phantom time from the
    ETA, collapsing a multi-candidate backlog toward zero.
  * A single finished candidate must not be multiplied out when a one-off tail dominates it. Box 1's
    first candidate cost 2.08 h, of which one 1800 s timeout was 0.83 h (40%) -- and its per-trial
    MEDIAN is 25.7 s against box 2's 30.6 s, i.e. it is the faster box. Charging every future
    candidate for that timeout turns "3.20 h more" into "5.70 h more", and 5.70 h against a 9.03 h
    remaining budget is the difference between "will reach Loop C" and "might not".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "scripts" / "project_loop_c_arrival.py"

_H = 3600.0


def _write(d: Path, events: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps({"seq": i, **e}) + "\n")
    return d


def _cand_cycle(t: float, cid: str, n_trials: int, trial_s: float,
                expand: bool, tail_s: float = 0.0) -> tuple[list[dict], float]:
    """One candidate's real event sequence, read off box 2's live log.

    publish -> prescreen -> N trials -> TUNING_DONE -> CLASSIFIED -> REPORTED
    -> [EXPANDED -> prescreen -> N trials -> TUNING_DONE -> CLASSIFIED -> REPORTED] -> STEP_DONE
    """
    evs: list[dict] = []

    def push(ts: float, typ: str, payload: dict) -> None:
        evs.append({"ts": ts, "type": typ, "payload": payload})

    cp = {"candidate_id": cid}
    push(t, "SPACE_PUBLISHED", {"space": {"space_id": "sp-" + cid, "candidate_id": cid}})
    t += 120.0
    push(t, "SPACE_PRESCREENED", cp)
    for i in range(n_trials):
        t += trial_s
        push(t, "TRIAL_DONE", {"trial": {"candidate_id": cid, "space_id": "sp-" + cid,
                                         "status": "complete", "failure_kind": None}})
    if tail_s:
        t += tail_s
        push(t, "TRIAL_DONE", {"trial": {"candidate_id": cid, "space_id": "sp-" + cid,
                                         "status": "fail", "failure_kind": "timeout"}})
    push(t, "TUNING_DONE", cp)
    push(t, "STATS_DONE", {})
    t += 10.0
    push(t, "BOTTLENECK_CLASSIFIED", cp)
    push(t, "DIMENSION_STATE", {"candidate_id": cid, "records": []})
    t += 300.0
    push(t, "BOTTLENECK_REPORTED", cp)
    if expand:
        t += 60.0
        push(t, "SPACE_PUBLISHED", {"space": {"space_id": "sp2-" + cid, "candidate_id": cid}})
        push(t, "SPACE_EXPANDED", cp)
        t += 180.0
        push(t, "SPACE_PRESCREENED", cp)
        for i in range(n_trials):
            t += trial_s
            push(t, "TRIAL_DONE", {"trial": {"candidate_id": cid, "space_id": "sp2-" + cid,
                                             "status": "complete", "failure_kind": None}})
        push(t, "TUNING_DONE", cp)
        push(t, "STATS_DONE", {})
        t += 10.0
        push(t, "BOTTLENECK_CLASSIFIED", cp)      # the SECOND one for this candidate
        push(t, "DIMENSION_STATE", {"candidate_id": cid, "records": []})
        t += 300.0
        push(t, "BOTTLENECK_REPORTED", cp)
    push(t, "STEP_DONE", {"step_key": "pipeline-" + cid})
    return evs, t


def _run(rd: Path, extra: list[str] | None = None) -> str:
    proc = subprocess.run([sys.executable, "-B", str(CHK), *(extra or []), "ARM", str(rd)],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return proc.stdout + proc.stderr


def _build(tmp: Path, name: str, cands: list[tuple[str, int, float, bool, float]],
           n_seeds: int = 4) -> Path:
    """`cands` entries: (cid, n_trials, trial_s, expand, tail_s). All n_seeds are registered at t~0
    exactly as `_generate_seeds` does, but only the listed ones do any work."""
    evs: list[dict] = [{"ts": 0.0, "type": "RUN_CREATED", "payload": {}},
                       {"ts": 60.0, "type": "BASELINE_DONE",
                        "payload": {"baseline": {"kind": "eager",
                                                 "latency_ms": {"median": 11.0}}}},
                       {"ts": 60.0, "type": "STEP_DONE", "payload": {"step_key": "baseline"}}]
    for i in range(n_seeds):
        evs.append({"ts": 120.0, "type": "CANDIDATE_REGISTERED",
                    "payload": {"candidate": {"candidate_id": "cand-%d" % i}}})
    evs.append({"ts": 120.0, "type": "STEP_DONE", "payload": {"step_key": "seeds"}})
    t = 180.0
    for cid, n, ts_, ex, tail in cands:
        block, t = _cand_cycle(t, cid, n, ts_, ex, tail)
        evs.extend(block)
    return _write(tmp / name, evs)


def _range_endpoints(out: str) -> tuple[float, float] | None:
    """The two endpoints off the `Report the RANGE lo-hi h` line.

    Parsing them from that one line rather than hunting the first float after each phrase: an
    earlier version keyed on "naive: if" and picked up the candidate's DURATION (0.79) instead of
    the projection (2.36), then declared the baseline broken because 0.86 was not below 0.79. The
    projector was right; the parser was reading a different quantity.
    """
    import re
    m = re.search(r"Report the RANGE\s+(\d+\.\d+)-(\d+\.\d+)\s*h", out)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _sc_expansion_not_double_counted(tmp: Path):
    """Two candidates, both expanded => 4 classifications but 2 done."""
    d = _build(tmp, "exp", [("cand-0", 20, 30.0, True, 0.0), ("cand-1", 20, 30.0, True, 0.0)])
    out = _run(d)
    if "candidates DONE 2 of 4" not in out:
        return "two expanded candidates were not counted as 2 done:\n%s" % out
    if "4 classifications" not in out:
        return "the double classification was not surfaced:\n%s" % out
    return None


def _sc_registration_is_not_work(tmp: Path):
    """Only one candidate does work; the other three are registered and idle."""
    d = _build(tmp, "reg", [("cand-0", 20, 30.0, True, 0.0)])
    out = _run(d)
    if "not started at all: 3" not in out:
        return "registered-but-unstarted candidates were treated as started:\n%s" % out
    if "in flight" in out:
        return "a finished-and-nothing-else run reported something in flight:\n%s" % out
    return None


def _sc_tail_not_extrapolated(tmp: Path):
    """One finished candidate whose cost is dominated by a single timeout."""
    # 20 trials at 30 s = 600 s, plus a 1800 s timeout => the tail is ~60% of the trial time.
    d = _build(tmp, "tail", [("cand-0", 20, 30.0, False, 1800.0)])
    out = _run(d)
    if "NOT PROJECTABLE" not in out:
        return "a single finished candidate was projected from as if it were a rate:\n%s" % out
    if "RANGE" not in out:
        return "a tail-dominated single duration was not reported as a range:\n%s" % out
    ends = _range_endpoints(out)
    if ends is None:
        return "could not parse the range endpoints from:\n%s" % out
    lo, hi = ends
    if not (lo < hi):
        return ("the tail-adjusted endpoint (%.2f) is not below the naive one (%.2f), so the "
                "adjustment did nothing" % (lo, hi))
    return None


def _sc_clean_candidate_no_range(tmp: Path):
    """A single finished candidate with NO dominant tail must not invent a range."""
    d = _build(tmp, "clean", [("cand-0", 20, 30.0, False, 0.0)])
    out = _run(d)
    if "NOT PROJECTABLE" not in out:
        return "n=1 was projected from as a rate:\n%s" % out
    if "RANGE" in out:
        return ("a candidate with no dominant tail was given a tail adjustment, which invents a "
                "spread that the data does not show:\n%s" % out)
    return None


def _sc_rounds_short_circuit(tmp: Path):
    d = _build(tmp, "rounds", [("cand-0", 20, 30.0, False, 0.0)])
    with (d / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": 999, "ts": 99999.0, "type": "FAMILY_ROUND_RECORDED",
                             "payload": {"family_id": "f1", "conversion": "improved"}}) + "\n")
    out = _run(d)
    if "Loop C has started" not in out:
        return "a run already in Loop C was still given an ETA to Loop C:\n%s" % out
    return None


SCENARIOS = [
    ("an expanded candidate is one candidate", _sc_expansion_not_double_counted),
    ("registration is not the start of work", _sc_registration_is_not_work),
    ("a tail-dominated duration yields a range", _sc_tail_not_extrapolated),
    ("a clean duration yields no range", _sc_clean_candidate_no_range),
    ("a run already in Loop C gets no ETA", _sc_rounds_short_circuit),
]

VARIANTS: list[tuple[str, str, str, list[str], str]] = [
    (
        "candidates counted by BOTTLENECK_CLASSIFIED instead of STEP_DONE",
        '        elif t == "STEP_DONE" and last_cand and last_cand in first_work:',
        '        elif t == "BOTTLENECK_CLASSIFIED" and last_cand and last_cand in first_work:',
        ["an expanded candidate is one candidate"],
        "the mis-count I actually made and reported: box 2's 4 classifications read as 4 candidates "
        "when they were 2, because classification fires again after the K expansion. It also halves "
        "the apparent per-candidate cost, since the first classification lands before the re-tune -- "
        "so the ETA is optimistic by roughly the whole re-tune",
    ),
    (
        "registration counted as the start of work",
        "            if t not in _REGISTRATION:\n"
        "                first_work.setdefault(cid, e[\"ts\"])",
        "            if True:\n"
        "                first_work.setdefault(cid, e[\"ts\"])",
        ["registration is not the start of work"],
        "`_generate_seeds` registers all four seeds at t~0, so every unstarted candidate reads as "
        "in flight since the beginning -- and that phantom elapsed time is SUBTRACTED from the ETA, "
        "collapsing a 3-candidate backlog toward zero. The friendly-looking direction: the run "
        "appears nearly done",
    ),
    (
        "the tail adjustment removed: a one-off timeout charged to every future candidate",
        '            if r["worst_trial_h"] and r["worst_trial_h"] > 0.15 * ds[0]:',
        "            if False:",
        ["a tail-dominated duration yields a range"],
        "measured on box 1: its one finished candidate cost 2.08 h of which one 1800 s timeout was "
        "0.83 h. The naive projection is 5.70 h more against a 9.03 h remaining budget; net of the "
        "tail it is 3.20 h. Box 1's per-trial MEDIAN is 25.7 s against box 2's 30.6 s -- it is the "
        "FASTER box -- so 5.70 h alone would have supported an intervention that the data does not",
    ),
    (
        "the tail adjustment always applied, even with no tail",
        '            if r["worst_trial_h"] and r["worst_trial_h"] > 0.15 * ds[0]:',
        "            if True:",
        ["a clean duration yields no range"],
        "invents a spread on a candidate whose trials were all typical, so every single-candidate "
        "projection reports a range and the range stops meaning 'a tail was found'. A qualifier "
        "that is always present carries no information",
    ),
    (
        "the Loop C short-circuit dropped",
        '    if r["rounds"]:',
        "    if False:",
        ["a run already in Loop C gets no ETA"],
        "a run that already produced FAMILY_ROUND_RECORDED still gets an ETA to its first one. "
        "Harmless-looking and actively misleading at the moment the checkpoint arrives, which is "
        "the one moment this script is consulted",
    ),
]


def main() -> int:
    original = CHK.read_text(encoding="utf-8")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("baseline:")
        base: dict[str, str | None] = {}
        for label, fn in SCENARIOS:
            err = fn(tmp / ("base_" + label.replace(" ", "_")[:28]))
            base[label] = err
            print("  %-44s %s" % (label, "ok" if err is None else "**FAILS**: " + err[:300]))
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
                    print("**SKIPPED** %s: does not parse (%s)" % (label, exc))
                    ok = False
                    continue
                CHK.write_text(patched, encoding="utf-8")
                try:
                    got = {l: fn(tmp / (label.replace(" ", "_")[:22] + "_"
                                        + l.replace(" ", "_")[:22]))
                           for l, fn in SCENARIOS}
                finally:
                    CHK.write_text(original, encoding="utf-8")
                survived = [s for s in must_break if got.get(s) is None]
                if not must_break:
                    print("NOEVID  %s" % label)
                elif survived:
                    ok = False
                    print("**FAIL** %s" % label)
                    print("        still passed on the broken projector: %s" % ", ".join(survived))
                else:
                    print("ok      %s" % label)
                    print("        %d/%d scenarios broke as required" % (
                        len(must_break), len(must_break)))
                print("        wrong version: %s" % why)
                print()
        finally:
            CHK.write_text(original, encoding="utf-8")
        for label, fn in SCENARIOS:
            if fn(tmp / ("after_" + label.replace(" ", "_")[:28])) is not None:
                print("!! RESTORE DID NOT COME BACK CLEAN -- check git status")
                return 2
    print("restored, baseline clean again")
    if not ok:
        print("\nVERDICT: at least one scenario is not evidence -- see above")
        return 1
    print("\nVERDICT: every scenario breaks on the projector it was written against")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
