"""Extract a real (best candidate source, BottleneckReport, ProfileRecord) triple from a finished run.

The G9 A/B has to start from something the harness actually produced and actually measured -- a
hand-written stand-in would test the stand-in. So this replays events.jsonl, finds the best candidate
by measured latency, and writes out the three files the A/B needs, plus the numbers so the choice is
auditable rather than asserted.
"""
import argparse
import io
import json
import sys
from pathlib import Path


def read_events(run_dir: Path):
    ev = []
    with io.open(run_dir / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = read_events(run)
    types = {}
    for e in ev:
        types[e.get("type", "?")] = types.get(e.get("type", "?"), 0) + 1
    print("events: %d, types: %s" % (len(ev), sorted(types.items(), key=lambda kv: -kv[1])[:12]))

    # Best trial by measured latency, and which candidate it belongs to.
    best = None
    for e in ev:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("status") != "complete":
            continue
        lat = t.get("latency_ms") or {}
        # The stored keys are `median`/`mean`, NOT `median_ms`/`robust_ms`. Guessing the
        # suffixed names read every trial as unmeasured and reported "no complete trial" on a
        # run holding 680 of them -- the same shape of error as reading payload.status when the
        # fields live under payload.trial.
        ms = lat.get("median") or lat.get("robust_ms") or lat.get("mean")
        if ms and (best is None or ms < best[0]):
            best = (ms, t.get("candidate_id"), t)   # candidate_id lives INSIDE trial
    if best is None:
        print("FAIL: no complete trial with a latency in this run")
        return 1
    ms, cand_id, trial = best
    print("best trial: %.4f ms  candidate=%s" % (ms, cand_id))

    # That candidate's report and profile.
    report = None
    for e in ev:
        if e.get("type") == "BOTTLENECK_REPORTED" and (e.get("payload") or {}).get("candidate_id") == cand_id:
            report = (e.get("payload") or {}).get("report")
    if report is None:
        for e in ev:
            if e.get("type") == "BOTTLENECK_REPORTED":
                report = (e.get("payload") or {}).get("report")
                cand_id = (e.get("payload") or {}).get("candidate_id")
        if report is not None:
            print("note: no report for the best candidate; using candidate %s instead" % cand_id)
    if report is None:
        print("FAIL: no BOTTLENECK_REPORTED event in this run")
        return 1

    profile = trial.get("profile")
    # Real layout: candidates/<candidate_id>/source.py -- a DIRECTORY per candidate, holding the
    # source plus its two witnesses and its trials. A `*.py` glob over candidates/ matches nothing
    # and reads as "no source on disk" on a run that has ten of them.
    src = run / "candidates" / (cand_id or "") / "source.py"
    if not src.exists():
        avail = sorted(d.name for d in (run / "candidates").iterdir() if d.is_dir())
        print("FAIL: no source.py for %s; candidates present: %s" % (cand_id, avail))
        return 1

    (out / "best.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if profile:
        (out / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print()
    print("source : %s -> %s" % (src.name, out / "best.py"))
    # A BottleneckReport is the ANALYST AGENT's prose (summary / hypotheses / suggested_action);
    # the harness classifier's structured verdict is a separate BOTTLENECK_CLASSIFIED event. Printing
    # `kind`/`evidence` here reported None/0 because those belong to the verdict, not the report.
    print("report : summary=%dch  hypotheses=%d  action=%dch"
          % (len(report.get("summary") or ""), len(report.get("hypotheses") or []),
             len(report.get("suggested_action") or "")))
    print("profile: %s" % ("yes" if profile else "MISSING (rich arm loses its extra fields)"))
    print("params : %s" % json.dumps(trial.get("params") or {})[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
