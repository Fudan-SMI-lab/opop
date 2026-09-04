#!/bin/bash
# Emit one line per interesting event across the L3 chain runs (43 now, 48 when it starts).
# Each stdout line becomes one Monitor notification.
cd /d/Pyhon_projects/opop/v2 || exit 1
declare -A SEEN
while true; do
  for d in runs/run-l3-43-20260904-093730 $(ls -d runs/run-l3-48-* 2>/dev/null); do
    f="$d/events.jsonl"
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    s=${SEEN[$f]:-}
    if [ -z "$s" ]; then
      # First sight of a NEW run: report from its beginning. For the run already in
      # flight when this monitor started, skip the backlog we have already analysed.
      case "$d" in
        *run-l3-43-20260904-093730) s=$n ;;
        *) s=0 ;;
      esac
    fi
    if [ "$n" -gt "$s" ]; then
      tail -n +$((s + 1)) "$f" | RF="$(basename "$d")" python scripts/fmt_events.py
      SEEN[$f]=$n
    fi
  done
  sleep 45
done
