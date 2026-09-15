"""Did the values S7 enqueued actually WIN their space, or only get measured?

WHY THIS EXISTS. `analyze_s7_pair.py` reports `points enqueued / refused` and, separately, that the
points were drawn -- and stops there. Those two facts together read as a success: the mechanism found a
wall, proposed a value, and the sampler measured it end to end. But "was measured" is not "was good",
and on the live S7 treatment arm the gap between the two is the entire result: all four enqueued points
were measured and NONE of them was the best value of its knob in its own space.

    space          knob              value  tail gain   best AT value   best ELSEWHERE
    sp-a92efa52    NUM_WARPS_QKV     16     34.14%      3.9700 (n=14)   3.7576 (n=7)     lost   5.7%
    sp-a92efa52    BM_QKV            256    30.37%      6.5249 (n=4)    3.7576 (n=17)    lost  74.0%
    sp-a92efa52    NUM_STAGES_QKV    3      50.78%      5.1144 (n=1)    3.7576 (n=20)    lost  36.0%
    sp-54660cd6    NUM_WARPS_QKV     8      55.15%      3.4467 (n=3)    3.2876 (n=14)    lost   4.8%

Reporting only the enqueue count and "it was drawn" would write that negative result up as a positive
one, which is why this check is a script and not a note: C2's premise is "a direction whose slope is
still steep and which a resource limit truncates is worth measuring", and the only reading that tests
the premise is whether the proposed value won.

WHAT IT MEASURES. For every point in `SLOPE_GUIDE_STEP.payload["enqueued"]`, within that point's own
space: the best median latency among trials that used the proposed knob value, against the best among
trials that used any OTHER value of the same knob. Comparison is WITHIN one space and one knob, because
this project has measured that pooling across candidates inverts correlations (Simpson's paradox on
register classes) and that a cross-dimension argmax tracks the normalisation scale rather than the
effect.

    python scripts/probes/did_the_enqueued_point_win.py <run_dir> [<run_dir> ...]
    python scripts/probes/did_the_enqueued_point_win.py --selftest

WHAT IT CANNOT DO, stated because a probe that overstates its reach is worse than none:

  * `n` is reported on both sides of every row and is often tiny. A row backed by one trial is not a
    finding -- the enqueued value may simply not have been retried. Rows are printed with their n and
    the summary counts only those meeting `--min-n` on BOTH sides.
  * A loss is not proof the wall was wrong. The proposed value can be genuinely better in a
    neighbourhood the sampler never reached with it, since the other knobs are not held fixed. This
    measures the value AS DRAWN, which is what the mechanism actually bought.
  * A win is not proof C2 works either: the enqueued value could be one TPE would have drawn anyway.
    `SLOPE_GUIDE_STEP` refuses already-drawn points, so a win means "not yet drawn when proposed", not
    "would never have been drawn".
"""

from __future__ import annotations

import argparse
import json
import sys
import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def _events(run_dir: Path) -> list[dict]:
    out: list[dict] = []
    path = run_dir / "events.jsonl"
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # An in-flight run's last line can be a partial write. Skipping it is right; failing
                # on it would make this probe unusable exactly when it is most wanted.
                continue
    return out


def _median_ms(trial: dict) -> float | None:
    """The trial's median latency, or None.

    `median`, NOT `median_ms`: the emitted keys are `median, mean, min, max, std, n_samples, samples`
    with no suffix, and reading the suffixed name returns None for EVERY trial -- which this project
    has already printed as the false fact "no comparable sibling in this family".
    """
    lat = trial.get("latency_ms")
    if not isinstance(lat, dict):
        return None
    val = lat.get("median")
    return float(val) if isinstance(val, (int, float)) else None


def _values(trial: dict) -> dict[str, Any]:
    params = trial.get("params")
    if not isinstance(params, dict):
        return {}
    vals = params.get("values")
    return vals if isinstance(vals, dict) else {}


def enqueued_points(events: list[dict]) -> list[dict]:
    """Every point S7 actually put into the sampler.

    Read from `SLOPE_GUIDE_STEP.payload["enqueued"]`, never by counting a `SLOPE_GUIDE_ENQUEUED`
    event: that name does not exist in src/, so counting it returns a credible constant zero. The
    emitter is orchestrator.py's `store.append("SLOPE_GUIDE_STEP", ...)` and the row fields come from
    `slope_guide.Suggestion.payload()` -- knob, knob_value, source, tail_gain_pct.
    """
    out: list[dict] = []
    for e in events:
        payload = e.get("payload") or {}
        space_id = str(payload.get("space_id") or "")
        if e.get("type") == "SLOPE_GUIDE_STEP":
            for row in payload.get("enqueued") or []:
                if not isinstance(row, dict):
                    continue
                out.append({
                    "space_id": str(row.get("space_id") or space_id),
                    "knob": str(row.get("knob") or ""),
                    "knob_value": row.get("knob_value"),
                    "source": str(row.get("source") or "?"),
                    "tail_gain_pct": row.get("tail_gain_pct"),
                })
        # THE v4.1 PATH. In active mode there is no SLOPE_GUIDE_STEP at all -- the v4
        # scanner replaces it -- so reading only the v3 event reports "0 points enqueued"
        # for a run that enqueued plenty, and the preregistered endpoint reads as absent
        # rather than as measured. Only E-kind blocks are enqueued VALUES in the sense this
        # probe tests: a C4 block re-measures two existing endpoints under frozen partners
        # (that is the source contrast, not a proposal), while E1/E2/E4 put a NEW value on
        # the axis, which is the thing whose winning-or-not is the endpoint.
        elif e.get("type") == "SCAN_BLOCK_ADMITTED" and payload.get("kind") in (
                "E1", "E2", "E4"):
            axis = str(payload.get("axis") or "")
            for row in payload.get("enqueued") or []:
                if not isinstance(row, dict):
                    continue
                raw = row.get("axis_value")
                if raw is None:
                    # Runs journalled before axis_value was added carry role/axis/direction
                    # only. Say so per point instead of scoring it: a missing value is not a
                    # loss, and silently dropping it would understate the enqueue count.
                    out.append({"space_id": space_id, "knob": axis, "knob_value": None,
                                "source": "v4.1:%s" % payload.get("kind"),
                                "tail_gain_pct": None,
                                "unreadable": "axis_value not journalled in this run"})
                    continue
                try:
                    value = ast.literal_eval(raw)      # written as repr(), so eval it back
                except (ValueError, SyntaxError):
                    value = raw
                out.append({
                    "space_id": space_id,
                    "knob": str(row.get("axis") or axis),
                    "knob_value": value,
                    "source": "v4.1:%s/%s" % (payload.get("kind"), row.get("direction")),
                    "tail_gain_pct": None,             # v4.1 carries an interval, not a scalar
                })
    return out


def trials_by_space(events: list[dict]) -> dict[str, list[dict]]:
    by_space: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("type") != "TRIAL_DONE":
            continue
        payload = e.get("payload") or {}
        trial = payload.get("trial") if isinstance(payload.get("trial"), dict) else payload
        # `status == "complete"` only: TRIAL_DONE fires for failures too, and a failed trial has no
        # latency. Counting all of them is what made two arms order oppositely on "trials".
        if trial.get("status") != "complete":
            continue
        space_id = str(trial.get("space_id") or payload.get("space_id") or "")
        by_space[space_id].append(trial)
    return by_space


def judge(point: dict, trials: list[dict]) -> dict:
    """Best latency AT the proposed value vs best at any OTHER value of the same knob."""
    knob, want = point["knob"], point["knob_value"]
    at: list[float] = []
    other: list[float] = []
    for t in trials:
        vals = _values(t)
        if knob not in vals:
            continue
        ms = _median_ms(t)
        if ms is None:
            continue
        # Compare as strings as well as by ==: a knob value can be journalled as 16 or "16" depending
        # on how the space declared it, and an == that silently fails would put every trial in
        # `other` and report a loss for a point that won.
        if vals[knob] == want or str(vals[knob]) == str(want):
            at.append(ms)
        else:
            other.append(ms)
    best_at = min(at) if at else None
    best_other = min(other) if other else None
    if point.get("unreadable"):
        # The value is not in the journal, so no trial can be matched to it. That is a
        # READABILITY gap in the emitter, not a measurement outcome -- reporting it as
        # `never_measured` would say the mechanism proposed a value nothing ever ran, which
        # is a claim about the mechanism rather than about the log. Measured live: four v4.1
        # E admissions, all readable-as-never_measured under the old wording.
        verdict = "value_not_journalled"
    elif best_at is None:
        verdict = "never_measured"
    elif best_other is None:
        verdict = "no_comparison"       # the only value of this knob ever drawn
    elif best_at < best_other:
        verdict = "won"
    elif best_at > best_other:
        verdict = "lost"
    else:
        verdict = "tied"
    delta = None
    if best_at is not None and best_other is not None and best_other > 0:
        delta = (best_at - best_other) / best_other * 100.0
    return {**point, "n_at": len(at), "n_other": len(other), "best_at": best_at,
            "best_other": best_other, "verdict": verdict, "delta_pct": delta}


def report(run_dir: Path, min_n: int, noise_pct: float) -> dict:
    events = _events(run_dir)
    points = enqueued_points(events)
    by_space = trials_by_space(events)
    rows = [judge(p, by_space.get(p["space_id"], [])) for p in points]

    print("=" * 100)
    print("run: %s" % run_dir)
    print("points enqueued by S7: %d   (spaces touched: %d)"
          % (len(rows), len({r["space_id"] for r in rows})))
    if not rows:
        # A structural zero and a broken read look identical from the outside, so say which this is.
        n_steps = sum(1 for e in events if e.get("type") == "SLOPE_GUIDE_STEP")
        n_failed = sum(1 for e in events if e.get("type") == "SLOPE_GUIDE_FAILED")
        print("nothing to judge. SLOPE_GUIDE_STEP events=%d, SLOPE_GUIDE_FAILED=%d" % (n_steps, n_failed))
        if n_steps == 0:
            print("zero steps => S7 is OFF in this arm (a control arm's structural zero), not a failure.")
        else:
            print("steps ran but enqueued nothing => walls were scarce or every point was already drawn;")
            print("read the step payloads' n_skipped_no_wall / n_skipped_already_proposed counters.")
        return {"n": 0, "won": 0, "lost": 0, "counted": 0}

    print()
    print("%-14s %-18s %-7s %-9s %-17s %-17s %s"
          % ("space", "knob", "value", "tailgain", "best AT value", "best ELSEWHERE", "verdict"))
    for r in sorted(rows, key=lambda r: (r["space_id"], r["knob"])):
        at = "%.4f (n=%d)" % (r["best_at"], r["n_at"]) if r["best_at"] is not None else "-- (n=0)"
        ot = "%.4f (n=%d)" % (r["best_other"], r["n_other"]) if r["best_other"] is not None else "-- (n=0)"
        tag = r["verdict"]
        if r["delta_pct"] is not None and r["verdict"] in ("won", "lost"):
            tag = "%s %.1f%%" % (r["verdict"], abs(r["delta_pct"]))
        thin = "" if min(r["n_at"], r["n_other"]) >= min_n else "   [thin: n<%d]" % min_n
        print("%-14s %-18s %-7s %-9s %-17s %-17s %s%s"
              % (r["space_id"][:14], r["knob"][:18], str(r["knob_value"])[:7],
                 ("%.2f%%" % r["tail_gain_pct"]) if isinstance(r["tail_gain_pct"], (int, float)) else "?",
                 at, ot, tag, thin))

    counted = [r for r in rows if min(r["n_at"], r["n_other"]) >= min_n]
    # A margin inside the re-evaluation noise floor is a TIE, not a loss, and printing 0.3% the same
    # way as 73.6% invites exactly the misreading it looks like: this project's independent re-eval gap
    # is +-2-4% with an UNSTABLE SIGN and the per-trial std is 16%, so a sub-noise margin does not
    # order the two values at all. Reported as its own bucket rather than folded into won/lost,
    # because "the mechanism picked a value that is no worse" is a different claim from either.
    def bucket(r: dict) -> str:
        if r["delta_pct"] is not None and abs(r["delta_pct"]) < noise_pct:
            return "tied"
        return r["verdict"]
    won = sum(1 for r in counted if bucket(r) == "won")
    lost = sum(1 for r in counted if bucket(r) == "lost")
    tied = sum(1 for r in counted if bucket(r) == "tied")
    print()
    print("of %d enqueued point(s), %d have >=%d trials on BOTH sides: %d won, %d lost, %d tied"
          % (len(rows), len(counted), min_n, won, lost, tied))
    sub = [r for r in counted if r["delta_pct"] is not None and abs(r["delta_pct"]) < noise_pct]
    if sub:
        print("   (%d of those are within the +-%.1f%% noise floor, so they are ties rather than "
              "losses: %s)" % (len(sub), noise_pct,
                               ", ".join("%s %+.1f%%" % (r["knob"][:14], r["delta_pct"]) for r in sub)))
    # Are two rows in the same space actually the same observation? Two points in one space can share
    # most of their trials -- measured on this run, `warps=8` and `bm=128` in sp-a9438715 each had 15
    # trials with 8 IN COMMON, and each row's best was the same trial. The verdicts are then coupled by
    # construction, and counting them as two independent losses overstates the evidence. Not a
    # double-count (the trial sets differ) so the rows are kept; the coupling is stated instead.
    for space in sorted({r["space_id"] for r in counted}):
        here = [r for r in counted if r["space_id"] == space]
        if len(here) < 2:
            continue
        same = [r for r in here if r["best_at"] is not None
                and abs(r["best_at"] - here[0]["best_at"]) < 1e-9
                and r["best_other"] is not None
                and abs(r["best_other"] - here[0]["best_other"]) < 1e-9]
        if len(same) >= 2:
            print()
            print("   note: %d rows in %s report IDENTICAL best-at/best-elsewhere (%s) -- their trial"
                  % (len(same), space, ", ".join(r["knob"][:16] for r in same)))
            print("   sets overlap, so these are ~1 observation, not %d. Do not count them separately."
                  % len(same))
    if counted and won == 0:
        print()
        print("!! NO ENQUEUED POINT WON ITS KNOB. The mechanism ran end to end -- wall found, point")
        print("   enqueued, point measured -- and the values it chose were not the good ones. That is a")
        print("   result ABOUT C2's premise, not a bug: reporting the enqueue count alone would present")
        print("   it as a success.")
    # Is slope magnitude worth anything as a ranking signal? A steep wall that loses badly while a
    # shallow one loses barely is the reading that matters, so print the pairing rather than a rho on
    # four points -- a correlation over n=4 would be a number with no content.
    ranked = [r for r in counted if isinstance(r["tail_gain_pct"], (int, float))
              and r["delta_pct"] is not None]
    if len(ranked) >= 2:
        ranked.sort(key=lambda r: -r["tail_gain_pct"])
        print()
        print("tail gain vs outcome, steepest first (does a steeper wall pick a better value?):")
        for r in ranked:
            print("    %6.2f%% slope -> %+7.1f%%  %s/%s" % (r["tail_gain_pct"], r["delta_pct"],
                                                            r["space_id"][:12], r["knob"][:16]))
    return {"n": len(rows), "won": won, "lost": lost, "counted": len(counted)}


# ---------------------------------------------------------------------------------------------------
# selftests. Every fixture is built from the EMITTER's field names -- orchestrator.py's
# store.append("SLOPE_GUIDE_STEP", {...}) and slope_guide.Suggestion.payload() -- not from what this
# reader would like to see. A fixture invented to match the reader proves only that the reader agrees
# with itself; this project has already shipped a green suite over three field names that do not exist.
# ---------------------------------------------------------------------------------------------------

def _sg(space: str, rows: list[dict]) -> dict:
    return {"type": "SLOPE_GUIDE_STEP",
            "payload": {"candidate_id": "c1", "space_id": space, "n_told": 10, "budget": 40,
                        "enqueued": rows, "refused": []}}


def _trial(space: str, values: dict, ms: float | None, status: str = "complete") -> dict:
    lat = {"median": ms, "mean": ms, "min": ms, "max": ms, "std": 0.0,
           "n_samples": 20, "samples": []} if ms is not None else None
    return {"type": "TRIAL_DONE",
            "payload": {"trial": {"space_id": space, "candidate_id": "c1", "status": status,
                                  "params": {"values": values}, "latency_ms": lat}}}


def _selftest() -> int:
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        print("  %-64s %s" % (name, "ok" if cond else "FAIL"))
        if not cond:
            fails.append(name)

    row16 = {"knob": "W", "knob_value": 16, "source": "soft_wall", "tail_gain_pct": 34.14}

    # 1. a losing point
    ev = [_sg("sp1", [row16]),
          _trial("sp1", {"W": 16}, 3.97), _trial("sp1", {"W": 8}, 3.7576)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a proposed value slower than another value reads 'lost'", r["verdict"] == "lost")
    check("the loss margin is relative to the better value", abs(r["delta_pct"] - 5.65) < 0.2)

    # 2. a winning point -- the positive control. Without this, a reader that always said "lost"
    #    would pass every other test here.
    ev = [_sg("sp1", [row16]),
          _trial("sp1", {"W": 16}, 3.10), _trial("sp1", {"W": 8}, 3.7576)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a proposed value faster than every other value reads 'won'", r["verdict"] == "won")

    # 3. failures must not count as measurements of the value
    ev = [_sg("sp1", [row16]),
          _trial("sp1", {"W": 16}, None, status="failed"), _trial("sp1", {"W": 8}, 3.7576)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a failed trial at the value does not count as measured",
          r["verdict"] == "never_measured" and r["n_at"] == 0)

    # 4. string/int mismatch must not fake a loss
    ev = [_sg("sp1", [{**row16, "knob_value": "16"}]),
          _trial("sp1", {"W": 16}, 3.10), _trial("sp1", {"W": 8}, 3.7576)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("knob value '16' matches a trial's 16 (no type-mismatch false loss)", r["verdict"] == "won")

    # 5. another space's trials must not be borrowed
    ev = [_sg("sp1", [row16]),
          _trial("sp1", {"W": 16}, 3.97), _trial("sp2", {"W": 8}, 1.0)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a faster trial in ANOTHER space is not compared against", r["verdict"] == "no_comparison")

    # 6. the event name that does not exist must not be the read path
    ev = [{"type": "SLOPE_GUIDE_ENQUEUED", "payload": {"space_id": "sp1", **row16}}]
    check("SLOPE_GUIDE_ENQUEUED events are NOT counted as enqueues", enqueued_points(ev) == [])

    # 7. reading the suffixed latency key must not be how this works
    ev = [_sg("sp1", [row16])]
    t = {"space_id": "sp1", "status": "complete", "params": {"values": {"W": 16}},
         "latency_ms": {"median": 3.97}}
    check("median is read from `median`, not `median_ms`", _median_ms(t) == 3.97)
    check("a payload carrying only median_ms reads as unmeasured",
          _median_ms({"latency_ms": {"median_ms": 3.97}}) is None)

    # 8. space_id on the row wins over the step's, since a K expansion re-spaces mid-pass
    ev = [_sg("spOUTER", [{**row16, "space_id": "spINNER"}])]
    check("a row's own space_id overrides the step's", enqueued_points(ev)[0]["space_id"] == "spINNER")

    # 9. structural zero is distinguishable from a broken read
    check("no SLOPE_GUIDE_STEP events => zero enqueued points, not an error",
          enqueued_points([_trial("sp1", {"W": 8}, 3.0)]) == [])

    # 10. tie
    ev = [_sg("sp1", [row16]), _trial("sp1", {"W": 16}, 3.5), _trial("sp1", {"W": 8}, 3.5)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("equal latencies read 'tied', not a win", r["verdict"] == "tied")

    # 11. the noise gate: a sub-noise margin is a tie, and a real loss must SURVIVE it. Both
    #     directions, because a gate that swallowed the 73.6% row would turn the headline result of
    #     this run into "all ties" -- a broken probe that reads as good news.
    ev = [_sg("sp1", [row16]), _trial("sp1", {"W": 16}, 3.6608), _trial("sp1", {"W": 8}, 3.6516)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a 0.3% margin is inside the 4% noise floor",
          r["verdict"] == "lost" and abs(r["delta_pct"]) < 4.0)
    ev = [_sg("sp1", [row16]), _trial("sp1", {"W": 16}, 6.5249), _trial("sp1", {"W": 8}, 3.7576)]
    r = judge(enqueued_points(ev)[0], trials_by_space(ev)["sp1"])
    check("a 73.6% loss is far outside the noise floor and stays a loss",
          r["verdict"] == "lost" and r["delta_pct"] > 4.0)

    print()
    print("%d/%d checks passed" % (14 - len(fails), 14))
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--min-n", type=int, default=2,
                    help="trials required on BOTH sides before a row counts in the summary (default 2)")
    ap.add_argument("--noise-pct", type=float, default=4.0,
                    help="margins smaller than this are ties, not wins/losses. Default 4.0 -- the top "
                         "of this project's measured re-eval gap (+-2-4%%, sign unstable)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.runs:
        ap.error("give at least one run dir, or --selftest")

    totals = {"n": 0, "won": 0, "lost": 0, "counted": 0}
    for run in args.runs:
        res = report(run, args.min_n, args.noise_pct)
        for k in totals:
            totals[k] += res[k]
        print()
    if len(args.runs) > 1:
        print("=" * 100)
        print("ACROSS %d RUN(S): %d enqueued, %d judged, %d won, %d lost"
              % (len(args.runs), totals["n"], totals["counted"], totals["won"], totals["lost"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
