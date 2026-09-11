#!/bin/bash
# Watch one in-flight experiment box and emit a line ONLY on something worth acting on.
#
# WHY THIS IS A FILE AND NOT AN INLINE COMMAND. The first version of these monitors used
# `pgrep -f "kernel_optimizer.cli"` over ssh, which MATCHES ITS OWN SSH COMMAND STRING: the
# pattern appears in the remote sshd's argv, so `pgrep` found itself, the until-loop condition was
# permanently true, and all three monitors could never fire. Proven rather than reasoned:
# `pgrep -f "kernel_optimizer.cli.NONEXISTENT_SUFFIX"` also returned 0. The bracket form
# `[k]ernel_optimizer.cli` cannot match the literal string in its own argv and is used everywhere
# below.
#
# COVERAGE -- silence must not be the only signal. A monitor that watches for a clean exit stays
# quiet through a crash, a hang, or an OOM, and that silence is indistinguishable from "still
# running fine". So this reports FOUR terminal-or-notable states, not one:
#
#   ENDED        the orchestrator process is gone (with the log tail, which says why)
#   STALLED      the process is alive but events.jsonl has not grown in STALL_MIN minutes --
#                the shape of an agent call hung on a read timeout, or a wedged ptxas
#   UNREACHABLE  4 CONSECUTIVE ssh failures. AutoDL ssh fails transiently under load (measured
#                load 13-19 during L3 runs), so a single failure is NOT a fault and must not
#                page; four in a row at this interval is ~16 min of no contact.
#   RECOVERED    contact or progress came back after one of the above, so a transient does not
#                leave a false alarm standing as the last word
#
# Usage: watch_run.sh <box1|box2|box3>
set -u

BOX="${1:?usage: watch_run.sh <box1|box2|box3>}"

case "$BOX" in
  box1) SSH=(ssh -o ConnectTimeout=60 autodl)
        EVENTS=/root/autodl-tmp/opop-workspace/opop-glm/runs-v3/run-l3-43-20260911-230217/events.jsonl
        LOG=/root/autodl-tmp/e1-box1.log
        LABEL="BOX1 E1-control  L3:43" ;;
  box2) SSH=(ssh -o ConnectTimeout=60 -p 22010 -i "$HOME/.ssh/autodl2_key" root@connect.bjb2.seetacloud.com)
        EVENTS=/root/autodl-tmp/opop-workspace/opop-glm/runs-v3/run-l3-43-20260911-230736/events.jsonl
        LOG=/root/autodl-tmp/e1-box2.log
        LABEL="BOX2 E1-treatment L3:43" ;;
  box3) SSH=(ssh -o ConnectTimeout=60 a800)
        EVENTS=/root/autodl-tmp/work/opop-glm/runs-l3/run-l3-48-20260911-231217/events.jsonl
        LOG=/root/autodl-tmp/e3-box3.log
        LABEL="BOX3 E3          L3:48" ;;
  *)    echo "unknown box: $BOX"; exit 2 ;;
esac

INTERVAL=240          # 4 min: an L3 trial takes minutes, so this cannot miss a phase
STALL_MIN=25          # a single agent call may legitimately run ~25 min (read timeout is 1500 s)
fails=0
last_size=""
stalled_since=""
reported_stall=0
reported_unreachable=0

# One remote round trip returns everything, so a poll costs one ssh rather than three.
probe() {
  "${SSH[@]}" "n=\$(pgrep -c -f '[k]ernel_optimizer.cli' 2>/dev/null || echo 0); \
               s=\$(stat -c %s '$EVENTS' 2>/dev/null || echo 0); \
               echo \"\$n \$s\"" 2>/dev/null
}

while true; do
  out="$(probe)"
  if [ -z "$out" ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 4 ] && [ "$reported_unreachable" -eq 0 ]; then
      echo "$LABEL: UNREACHABLE -- 4 consecutive ssh failures (~16 min no contact). The run may still be fine; check by hand."
      reported_unreachable=1
    fi
    sleep "$INTERVAL"; continue
  fi
  if [ "$fails" -ge 4 ] && [ "$reported_unreachable" -eq 1 ]; then
    echo "$LABEL: RECOVERED -- ssh contact is back."
    reported_unreachable=0
  fi
  fails=0

  nproc="${out%% *}"
  size="${out##* }"

  # The process is gone: report and stop. This is the one terminal state.
  if [ "${nproc:-0}" -eq 0 ]; then
    echo "$LABEL: ENDED -- orchestrator process is gone. events.jsonl = $size bytes. Log tail:"
    "${SSH[@]}" "tail -6 '$LOG' 2>/dev/null" 2>/dev/null
    exit 0
  fi

  # Alive. Is it making progress? events.jsonl is append-only, so its size is a monotone clock.
  if [ "$size" = "$last_size" ]; then
    if [ -z "$stalled_since" ]; then stalled_since=$(date +%s); fi
    quiet=$(( ($(date +%s) - stalled_since) / 60 ))
    if [ "$quiet" -ge "$STALL_MIN" ] && [ "$reported_stall" -eq 0 ]; then
      echo "$LABEL: STALLED -- alive but events.jsonl has not grown in ${quiet} min (size $size). Usually a hung agent call; the harness recovers from those, so this is a heads-up, not a failure."
      reported_stall=1
    fi
  else
    if [ "$reported_stall" -eq 1 ]; then
      echo "$LABEL: RECOVERED -- events are flowing again (size $size)."
    fi
    stalled_since=""
    reported_stall=0
  fi
  last_size="$size"
  sleep "$INTERVAL"
done
