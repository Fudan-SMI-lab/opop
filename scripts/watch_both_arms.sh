#!/bin/bash
# Follow whatever run is CURRENTLY writing, across both experiment arms.
#
# Replaces watch_l3_chain.sh / watch_l3_rerun.sh, which pinned specific run directories by
# date (`runs/run-l3-21-20260905-*`, `runs/run-l3-43-20260904-093730`). Both go blind the
# moment the queue advances: the GLM arm writes to ../v2-glm/runs, and the gpt runs that
# follow are dated 20260906. A watcher that names dates cannot survive a handover.
#
# This discovers run directories by GLOB over both arms' runs_dir and picks up new ones as
# they appear, so it keeps reporting through glm:21 -> gpt:43 -> gpt:48 without edits.
#
# A new directory is reported from its FIRST event, because for a run that starts after the
# watcher does, the backlog IS the news (baseline, seeds, first tunings). Directories already
# non-empty at startup are skipped to their current end so an already-analysed run does not
# replay hundreds of lines.
cd /d/Pyhon_projects/opop/v2 || exit 1

GPT_RUNS="runs"
GLM_RUNS="/d/Pyhon_projects/opop/v2-glm/runs"

declare -A SEEN
FIRST_PASS=1

while true; do
  for d in $(ls -d "$GPT_RUNS"/run-* "$GLM_RUNS"/run-* 2>/dev/null); do
    f="$d/events.jsonl"
    [ -f "$f" ] || continue
    n=$(wc -l < "$f" 2>/dev/null) || continue
    if [ -z "${SEEN[$f]:-}" ]; then
      # Pre-existing runs: skip their backlog. Runs that appear later: report from line 1.
      if [ "$FIRST_PASS" = "1" ]; then SEEN[$f]=$n; else SEEN[$f]=0; fi
    fi
    s=${SEEN[$f]}
    if [ "$n" -gt "$s" ]; then
      # Label with the arm so glm and gpt lines are never confused.
      case "$d" in
        "$GLM_RUNS"/*) arm="glm" ;;
        *)             arm="gpt" ;;
      esac
      tail -n +$((s + 1)) "$f" | RF="$arm:$(basename "$d")" python scripts/fmt_events.py
      SEEN[$f]=$n
    fi
  done
  FIRST_PASS=0
  sleep 45
done
