#!/bin/bash
# Watch the S7 pair: one line per change, plus a loud line if an arm dies.
#
# THREE TRAPS THIS AVOIDS, each of which bit a version of this watcher within one tick:
#
# 1. `grep -c PATTERN file || echo 0` prints "0\n0" on no match. `grep -c` already prints 0 AND exits
#    non-zero, so the fallback fires on top of the real answer -- the recorded
#    `a-fallback-can-turn-a-zero-count-into-a-positive` trap, in the direction that makes every field
#    two lines. Counting is done in python here, which returns one number and does not signal "no match"
#    through its exit code.
#
# 2. `pgrep -c -f kernel_optimizer` counts the WORKERS too: every GPU job's argv contains
#    ".../src/kernel_optimizer/gpu/worker_main.py", so the count reads 4 with two arms running and the
#    "an arm died" test never fires. Matched on the orchestrator's own argv (`kernel_optimizer.cli`)
#    instead, with `[k]` so the pgrep pattern cannot match its own command line over ssh.
#
# 3. Counting `RESOURCE_WALL_ATTRIBUTED` / `RESOURCE_SOFT_WALL` EVENTS as "walls". Both are emitted
#    unconditionally to record that the search ran, so an arm that found nothing still gets one of
#    each. Live reading: `walls=1 soft=1` on the treatment arm whose payloads were `"walls": []`,
#    `"walls_found": 0` and `applicable: false` -- a plausible number saying the opposite of the truth,
#    the `a-constant-reading-is-a-broken-probe` shape. Now the WALLS INSIDE the payload are counted.
set -u
cd /root/autodl-tmp/opop-workspace/opop-glm/runs-v3 || exit 1
PY=/root/autodl-tmp/orch-venv/bin/python

count() {  # count(file, type) -- exactly one number on stdout, 0 when absent
  "$PY" - "$1" "$2" << 'PYEOF'
import json, sys
path, want = sys.argv[1], sys.argv[2]
n = 0
try:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                if json.loads(line).get("type") == want:
                    n += 1
            except Exception:
                pass
except OSError:
    pass
print(n)
PYEOF
}

enqueued() {  # total points S7 actually put on the queue, and how many the tuner refused
  "$PY" - "$1" << 'PYEOF'
import json, sys
acc = ref = 0
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "SLOPE_GUIDE_STEP":
                continue
            p = e.get("payload") or e
            acc += len(p.get("enqueued") or [])
            ref += len(p.get("refused") or [])
except OSError:
    pass
print(f"{acc}/{ref}")
PYEOF
}

walls_found() {  # walls actually FOUND / soft-wall scans that were APPLICABLE -- not event counts
  "$PY" - "$1" << 'PYEOF'
import json, sys
hard = soft = 0
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            p = e.get("payload") or e
            if t == "RESOURCE_WALL_ATTRIBUTED":
                hard += len(p.get("walls") or [])
            elif t == "RESOURCE_SOFT_WALL":
                soft += len(p.get("walls") or [])
except OSError:
    pass
print(f"{hard}/{soft}")
PYEOF
}

prev=""
while true; do
  cur=""
  for arm in s7-treatment s7-control; do
    R=$(ls -d "$arm"/run-l3-43-*/ 2>/dev/null | tail -1)
    [ -n "$R" ] || continue
    E="$R/events.jsonl"
    [ -f "$E" ] || continue
    tr=$(count "$E" TRIAL_DONE)
    sg=$(count "$E" SLOPE_GUIDE_STEP)
    sgf=$(count "$E" SLOPE_GUIDE_FAILED)
    # TRAP 3, hit live: `RESOURCE_WALL_ATTRIBUTED` and `RESOURCE_SOFT_WALL` are emitted
    # UNCONDITIONALLY to record that the search ran -- the treatment arm's carried
    # `"walls": [], "walls_found": 0` and its soft scan `applicable: false`. Counting the EVENTS
    # therefore printed `walls=1 soft=1` for an arm that found nothing, which reads as the opposite
    # of the truth at exactly the moment the reading matters. Count the WALLS inside them instead.
    wf=$(walls_found "$E")
    fin=$(count "$E" RUN_FINISHED)
    eq=$(enqueued "$E")
    cur="$cur | ${arm#s7-} trials=$tr wallsfound(h/s)=$wf s7steps=$sg enq/ref=$eq s7fail=$sgf done=$fin"
  done
  alive=$(pgrep -cf "[k]ernel_optimizer.cli" || true)
  [ -z "$alive" ] && alive=0

  if [ "$cur" != "$prev" ]; then
    echo "alive=$alive$cur"
    prev="$cur"
  fi

  if [ "$alive" -lt 2 ]; then
    echo "ARM DOWN (alive=$alive)$cur"
    for n in treatment control; do
      L=/root/autodl-tmp/s7-$n.log
      [ -f "$L" ] || continue
      hit=$(grep -Ei "Traceback|Error|FAILED|assertion|Killed|OOM|No module" "$L" | tail -2)
      [ -n "$hit" ] && echo "  $n: $hit"
    done
    [ "$alive" -eq 0 ] && { echo "BOTH ARMS ENDED$cur"; break; }
  fi
  sleep 300
done
