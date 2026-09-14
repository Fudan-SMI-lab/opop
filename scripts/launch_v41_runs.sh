#!/bin/bash
# Launch one or more v4.1 orchestrator runs on one box, one per GPU, simultaneously.
#
# WHY A GENERAL LAUNCHER RATHER THAN A COPY PER EXPERIMENT. The two predecessors
# (launch_s7_pair.sh, launch_s4_c2_pair.sh) each hardcode their own arm names, configs, GPU
# indices, log paths and XDG dirs. That is exactly the shape that produced this project's
# worst class of bug: a probe hardcoded to one experiment's names reported the NEXT
# experiment's own workers as a foreign tenant, and the correct response to a false alarm
# ("restart the 12 h pair") is more expensive than no check at all. Everything that varies
# per experiment is an argument here; nothing about N1/M1/P1/P2 appears in this file.
#
# It handles both shapes window 1 needs:
#   * a PAIR on one box (two arms, same task, comparability-audited)  -- box4 N1
#   * two INDEPENDENT single-arm runs on one box (different tasks)    -- box1 P1 and P2
#
# Usage:
#   scripts/launch_v41_runs.sh --tag <tag> \
#       --job <label>,<config>,<cvd>,<task> [--job ...] \
#       [--pair <labelA>:<labelB> [--expect <dotted.key>]...]
#
#   --tag      names the log/pinning files under /root (one namespace per launch)
#   --job      label (log/journal name), config path, CUDA_VISIBLE_DEVICES, KernelBench task
#   --pair     two job labels that form a controlled pair => run the comparability audit
#              BEFORE spending the GPU hours. Omit for independent runs.
#   --expect   a config key ALLOWED to differ between the paired arms (repeatable).
#              For an A/A pair pass none: the arms must differ ONLY in isolation paths.
#
# Example (box4 N1 A/A pair):
#   scripts/launch_v41_runs.sh --tag n1 \
#     --job a,configs/experiments_v41_n1_a_box4gpu0.yaml,0,level3:43 \
#     --job b,configs/experiments_v41_n1_b_box4gpu1.yaml,1,level3:43 \
#     --pair a:b
#
# Example (box1 two independent pilots):
#   scripts/launch_v41_runs.sh --tag pilot \
#     --job p1,configs/experiments_v41_pilot_p1_box1gpu0.yaml,0,level3:43 \
#     --job p2,configs/experiments_v41_pilot_p2_box1gpu1.yaml,1,level3:21
set -uo pipefail

W=${OPOP_WORK:-/root/autodl-tmp/work/opop}
PY=${OPOP_PY:-/root/autodl-tmp/orch-venv/bin/python}

TAG=""
JOBS=()
PAIR=""
EXPECT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)    TAG=$2; shift 2 ;;
    --job)    JOBS+=("$2"); shift 2 ;;
    --pair)   PAIR=$2; shift 2 ;;
    --expect) EXPECT+=(--expect "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "--tag is required" >&2; exit 2; }
[ ${#JOBS[@]} -gt 0 ] || { echo "at least one --job is required" >&2; exit 2; }

cd "$W" || { echo "cannot cd to $W" >&2; exit 2; }

# Parse the job specs BEFORE touching the GPU, so a typo costs nothing.
LABELS=() CONFIGS=() CVDS=() TASKS=()
for spec in "${JOBS[@]}"; do
  IFS=, read -r lbl cfg cvd task <<< "$spec"
  [ -n "${task:-}" ] || { echo "malformed --job '$spec' (need label,config,cvd,task)" >&2; exit 2; }
  [ -f "$cfg" ] || { echo "missing config: $cfg" >&2; exit 2; }
  LABELS+=("$lbl"); CONFIGS+=("$cfg"); CVDS+=("$cvd"); TASKS+=("$task")
done

# Two jobs on the same physical card would time-slice each other's SMs/L2/bandwidth and
# contaminate every latency in both. Caught here, not discovered at wrap-up when the answer
# is gone.
dupes=$(printf '%s\n' "${CVDS[@]}" | sort | uniq -d)
if [ -n "$dupes" ]; then
  echo "REFUSING TO START: two jobs share CUDA_VISIBLE_DEVICES=$dupes" >&2
  exit 4
fi

# `[k]ernel_optimizer` so this grep cannot match its own command line. The count decides
# whether it is safe to start and a self-match can only INFLATE it, which is the safe
# direction; do not loosen the pattern.
running=$(ps -eo args | grep -c "[k]ernel_optimizer.cli" || true)
if [ "$running" != "0" ]; then
  echo "REFUSING TO START: $running orchestrator process(es) already running." >&2
  echo "These runs need their cards to themselves. Wait, or stop the current runs." >&2
  ps -eo pid,etimes,args | grep "[k]ernel_optimizer.cli" >&2
  exit 3
fi

# COMPARABILITY, checked before spending 12 h rather than discovered afterwards. Only for a
# declared pair: two independent single-arm runs on different tasks are not comparable by
# construction and auditing them would be meaningless.
if [ -n "$PAIR" ]; then
  IFS=: read -r pa pb <<< "$PAIR"
  cfg_a="" cfg_b=""
  for i in "${!LABELS[@]}"; do
    [ "${LABELS[$i]}" = "$pa" ] && cfg_a=${CONFIGS[$i]}
    [ "${LABELS[$i]}" = "$pb" ] && cfg_b=${CONFIGS[$i]}
  done
  [ -n "$cfg_a" ] && [ -n "$cfg_b" ] || { echo "--pair '$PAIR' names an unknown label" >&2; exit 2; }
  AUDIT=/root/$TAG-preaudit.txt
  if ! PYTHONPATH="$W/src" "$PY" scripts/audit_arm_comparability.py \
        "$cfg_a" "$cfg_b" "${EXPECT[@]+"${EXPECT[@]}"}" > "$AUDIT" 2>&1; then
    echo "REFUSING TO START: config audit failed. See $AUDIT" >&2
    tail -25 "$AUDIT" >&2
    exit 5
  fi
  echo "config audit ($pa vs $pb): $(grep -c 'THE VARIABLE' "$AUDIT") intended difference(s), COMPARABLE"
fi

# runs_dir and XDG_DATA_HOME come from the CONFIG, read through load_config -- not grepped
# out of the YAML and not retyped on the command line. A launch-env XDG that disagreed with
# the config's server_env would give the orchestrator and its opencode server different
# state dirs, and retyping is how two arms end up sharing one.
read_cfg() {
  PYTHONPATH="$W/src" "$PY" - "$1" <<'PYEOF'
import sys
from pathlib import Path
from kernel_optimizer.config import load_config
cfg = load_config(Path(sys.argv[1]))
xdg = (cfg.opencode.server_env or {}).get("XDG_DATA_HOME", "")
print(cfg.run.runs_dir)
print(xdg)
PYEOF
}

PIDS=() RUNDIRS=()
for i in "${!LABELS[@]}"; do
  lbl=${LABELS[$i]}; cfg=${CONFIGS[$i]}; cvd=${CVDS[$i]}; task=${TASKS[$i]}
  mapfile -t got < <(read_cfg "$cfg")
  runs_dir=${got[0]:-}; xdg=${got[1]:-}
  if [ -z "$runs_dir" ] || [ -z "$xdg" ]; then
    echo "REFUSING TO START: could not read runs_dir/XDG_DATA_HOME from $cfg" >&2
    exit 6
  fi
  mkdir -p "$runs_dir" "$xdg"
  RUNDIRS+=("$runs_dir")
  echo "starting $lbl on GPU $cvd  task=$task  runs_dir=$runs_dir"
  # setsid so the run's session leader is not this shell: an ssh disconnect, a local
  # reboot of the operator's machine, or this script exiting cannot take a 12 h run down.
  CUDA_VISIBLE_DEVICES="$cvd" XDG_DATA_HOME="$xdg" PYTHONPATH=src \
    setsid nohup "$PY" -m kernel_optimizer.cli \
    --config "$cfg" run --task "$task" \
    > "/root/$TAG-$lbl.log" 2>&1 &
  PIDS+=("$!")
done

echo
for i in "${!LABELS[@]}"; do
  echo "${LABELS[$i]}: pid=${PIDS[$i]} gpu=${CVDS[$i]} log=/root/$TAG-${LABELS[$i]}.log"
done

sleep 25
for i in "${!PIDS[@]}"; do
  if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
    echo "WARNING: ${LABELS[$i]} (pid ${PIDS[$i]}) is gone within 25 s -- read its log" >&2
    tail -15 "/root/$TAG-${LABELS[$i]}.log" >&2
  fi
done
ps -eo pid,etimes,args | grep "[k]ernel_optimizer.cli"

# WHICH CARD IS EACH RUN ACTUALLY ON. This is the ONLY time the question can be answered:
# GPU work is a one-shot subprocess per job, nothing on disk records the device (5689 job
# files across the S7 pair carry no CVD, no device index, no UUID), and after the runs end
# the answer is gone for good. Run at wrap-up the probe correctly printed "NO COMPUTE
# PROCESSES SEEN ... ANSWERS NOTHING" -- refusing to read its own silence as separation.
#
# /proc/<pid>/environ is the attribution: a per-process CVD read, not an inference from
# which card looks busy. Retried because the opening phase is an LLM call of unpredictable
# length -- the first attempt against the step-4 pair saw no GPU work at all at +7 min.
PIN=/root/$TAG-gpu-pinning.txt
PROBE=${OPOP_PROBE_DIR:-/root/probe-clean}/gpu_pinning_check.py
( echo "=== per-process CUDA_VISIBLE_DEVICES (read from /proc, the authoritative tie) ==="
  for p in $(pgrep -f "[k]ernel_optimizer.cli"); do
    cvd=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' || echo "CVD=(unset)")
    cfg=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -o 'experiments_[a-z0-9_]*' | head -1)
    echo "pid=$p $cvd cfg=$cfg"
  done
  echo
  if [ ! -f "$PROBE" ]; then
    echo "!! $PROBE missing -- the live separation probe DID NOT RUN. The /proc lines above"
    echo "!! are the only attribution; they are per-process and sufficient for CVD, but the"
    echo "!! physical-index cross-check is absent. Do not read this file as a clean verdict."
  else
    for attempt in $(seq 1 12); do
      echo "--- pinning attempt $attempt ($(date -u +%H:%M:%SZ)) ---"
      if "$PY" "$PROBE" $(printf '%s/* ' "${RUNDIRS[@]}") 2>&1; then
        echo "--- answered on attempt $attempt ---"
        break
      fi
      echo "--- attempt $attempt answered nothing (no GPU processes in window); retry in 10 min ---"
      sleep 600
    done
  fi
) > "$PIN" 2>&1 &

echo
echo "GPU separation check is running in the background -> $PIN  -- READ IT."
echo "If two runs share a card, restart them: contaminated latencies cannot be repaired"
echo "downstream, and once the runs end the question is unanswerable (no device on disk)."
