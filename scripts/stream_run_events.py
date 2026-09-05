#!/usr/bin/env python3
"""Stream selected events from a run's events.jsonl as one line per event.

Reads JSON lines on stdin (from `tail -f`) and prints a compact summary for the
event types worth a notification. Kept as a FILE rather than `python -c` inline:
an f-string cannot contain escaped double quotes, so the inline form is a
SyntaxError under 3.12 and the monitor dies silently at startup.

Usage:  tail -f -n +1 --retry <run>/events.jsonl | python scripts/stream_run_events.py [label]
"""
from __future__ import annotations

import json
import sys

WATCH = {
    "TUNING_DONE", "FAMILY_ROUND_RECORDED", "RUN_FINISHED", "SPACE_REJECTED",
    "SPACE_EXPANSION_REJECTED", "EXPANSION_CONSTRAINTS_RESTORED",
    "CANDIDATE_REGISTERED", "AGENT_CALL_FAILED", "KERNELS_NEVER_LAUNCHED",
    "CONVERGENCE_DECIDED",
}


def describe(kind: str, p: dict) -> str | None:
    if kind == "TUNING_DONE":
        return (f"TUNED {p.get('candidate_id')} {p.get('best_ms')} "
                f"improved={p.get('improved_family')}")
    if kind == "FAMILY_ROUND_RECORDED":
        return (f"ROUND {p.get('family_id')} best={p.get('best_ms')} "
                f"round={p.get('round')}")
    if kind == "EXPANSION_CONSTRAINTS_RESTORED":
        return (f"CONSTRAINT-RESTORE FIRED {p.get('candidate_id')}: "
                f"{p.get('restored')}")
    if kind in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED"):
        return f"{kind} {p.get('candidate_id')} {p.get('reason')}"
    if kind == "CANDIDATE_REGISTERED":
        c = p.get("candidate") or p
        if c.get("origin") == "seed":
            return f"SEED {c.get('candidate_id')} backend={c.get('backend')}"
        return None
    if kind == "AGENT_CALL_FAILED":
        return f"AGENT_FAILED {p.get('module')} {str(p.get('error'))[:150]}"
    if kind == "KERNELS_NEVER_LAUNCHED":
        return (f"DEAD KERNELS {p.get('candidate_id')} {p.get('never_launched')} "
                f"over {p.get('n_trials_measured')} trials")
    if kind == "CONVERGENCE_DECIDED":
        d = p.get("decision") or {}
        if d.get("verdict") != "freeze":
            return None
        return (f"FREEZE {p.get('family_id') or d.get('scope')} "
                f"stop={d.get('stop_kind')} "
                f"hist={(d.get('evidence') or {}).get('best_history')}")
    if kind == "RUN_FINISHED":
        b = (p.get("summary") or {}).get("best") or {}
        return (f"RUN_FINISHED tuned={b.get('tuned_ms')} "
                f"reeval={b.get('final_reeval_ms')} "
                f"verdict={json.dumps(b.get('honest_verdict'))}")
    return None


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    prefix = f"{label} " if label else ""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        kind = e.get("type")
        if kind not in WATCH:
            continue
        msg = describe(kind, e.get("payload") or {})
        if msg:
            print(prefix + msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
