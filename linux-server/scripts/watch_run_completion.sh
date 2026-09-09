#!/usr/bin/env bash
# Record a run's completion ON THE BOX, so a finish that happens with no session attached is
# still captured with its numbers.
#
# Every Monitor task lives in the local CLI, so a local reboot silences all of them. The runs
# themselves are detached (PPID 1, no controlling terminal) and carry on, and events.jsonl is
# the authoritative record either way -- but the post-run summary is worth capturing at the
# moment it happens rather than reconstructing later, and a one-line completion stamp makes
# "did it end on its own clock or stop early" answerable at a glance.
#
# Detaches itself with setsid + nohup so it outlives the ssh session that starts it.
#
# Usage: watch_run_completion.sh <run_dir> [label]
set -u

RUN="${1:?usage: watch_run_completion.sh <run_dir> [label]}"
LABEL="${2:-$(basename "$RUN")}"
VENV=/root/autodl-tmp/orch-venv
OUT="$RUN/../completion-$LABEL.txt"

{
    echo "=== watch armed $(date -Is) for $RUN"
    while /root/orchestrator_running.sh; do
        sleep 120
    done
    echo "=== orchestrator exited $(date -Is)"
    # RUN_FINISHED is written by _finalize() inside the orchestrator, so by the time the process
    # is gone the re-eval and the report are already on disk.
    if grep -q RUN_FINISHED "$RUN/events.jsonl" 2>/dev/null; then
        echo "RUN_FINISHED present -- the run completed its own finalize"
    else
        echo "WARNING: no RUN_FINISHED -- the orchestrator died before finalizing"
    fi
    echo
    "$VENV/bin/python" /root/run_summary.py "$RUN" 2>&1 | head -40
    echo
    echo "--- near-tie check (is the winner distinguishable?)"
    "$VENV/bin/python" /root/audit_ties.py "$RUN" 2>&1 | tail -12
    echo
    echo "--- report path"
    ls -la "$RUN/report/report.md" 2>/dev/null || echo "  no report.md"
} > "$OUT" 2>&1 &

echo "armed; will write $OUT"
