#!/bin/bash
# Emit one line per interesting event for the newest L3 run (48 rerun, then 21/43).
# Each stdout line becomes one Monitor notification.
cd /d/Pyhon_projects/opop/v2 || exit 1
declare -A SEEN
while true; do
  for d in $(ls -d runs/run-l3-48-20260905-* runs/run-l3-21-20260905-* \
                   runs/run-l3-43-20260905-* 2>/dev/null); do
    f="$d/events.jsonl"
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    s=${SEEN[$f]:-0}
    if [ "$n" -gt "$s" ]; then
      tail -n +$((s + 1)) "$f" | RF="$(basename "$d")" python scripts/fmt_events.py
      SEEN[$f]=$n
    fi
  done
  sleep 45
done
