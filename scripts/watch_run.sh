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
#
# ONE MONITOR PER BOX. Raising STALL_MIN in f9788c3 meant starting a fresh monitor, and the old
# one was left running -- so box 1 had two, the 04:55 copy still reading STALL_MIN=25 and paging
# on a condition that had already been diagnosed and decided. Found only because the negative that
# suggested the opposite (`ps aux | grep watch_run` returning 0) was itself wrong: Git-Bash's `ps`
# cannot see processes outside its own MSYS tree, so on Windows a process inventory must come from
# WMI (`Get-CimInstance Win32_Process | Where CommandLine -like '*watch_run*'`), never from `ps`.
# The guard below makes the duplicate impossible rather than relying on remembering.
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

# 40 min, raised from 25. A single job may legitimately occupy the box for its whole 1800 s (30 min)
# deadline without emitting an event, because `TRIAL_DONE` is written only when the job returns --
# and D8 is exactly that case: box 1's cand-941ea454 drives ptxas for 13-30 min per trial, so a
# 25 min threshold paged every ~28 min for a condition already diagnosed and decided (let it run).
# A monitor that fires repeatedly on a known state trains its reader to ignore it, which is worse
# than one that fires slightly late. 40 min sits above the 30 min job ceiling plus a margin, so
# anything it reports is genuinely outside the harness's own bounds.
STALL_MIN=40

# Refuse to be the second monitor on this box. A PID file is the only mechanism available here that
# works on Windows too, because `pgrep` cannot see a sibling started by a different Git-Bash
# invocation (that blindness is what let the duplicate live for 105 min unnoticed). The stale-file
# case is handled by rewriting it: if the recorded PID is gone, this instance takes over.
#
# SCOPE, measured rather than assumed: the PID written is `$$`, an MSYS pid, and `kill -0` reads MSYS
# pids -- so the guard is self-consistent between two runs of THIS script. It does NOT see a monitor
# whose pid you only know from WMI: the three leaf monitors alive on 2026-09-12 had Windows pids
# 45756 / 55792 / 23984 and `kill -0` reported all three dead. So do not backfill this file from a
# WMI listing -- a pid `kill -0` cannot see gives the weak failure (a duplicate still allowed), and
# one that collides with an unrelated live process gives the dangerous one (a legitimate restart
# refused). Before starting a monitor, still confirm the inventory with WMI; this guard is the
# backstop for the case that actually bit, which is starting a second copy from a second shell.
PIDFILE="${TMPDIR:-/tmp}/watch_run.$BOX.pid"
if [ -f "$PIDFILE" ]; then
  prev="$(cat "$PIDFILE" 2>/dev/null || true)"
  # `kill -0` tests liveness without signalling. Quoted and defaulted so an empty or garbage file
  # cannot expand into `kill -0` with no argument, which would succeed and wrongly refuse to start.
  if [ -n "${prev:-}" ] && kill -0 "$prev" 2>/dev/null; then
    echo "$LABEL: REFUSING TO START -- monitor pid $prev is already watching $BOX." >&2
    echo "  Stop it first (TaskStop, or kill $prev). Two monitors on one box means the older one" >&2
    echo "  keeps paging with whatever thresholds it was started with -- exactly the f9788c3 case." >&2
    exit 3
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT INT TERM

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
