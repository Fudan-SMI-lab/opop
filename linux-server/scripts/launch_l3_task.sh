#!/usr/bin/env bash
# Launch ONE L3 task on this box, refusing to start if another orchestrator holds the GPU.
#
# Usage: launch_l3_task.sh <task> [config]
#   e.g. launch_l3_task.sh level3:21
#
# Why a script rather than an inline ssh command: the "is an orchestrator already running"
# guard has to not match ITSELF. `pgrep -f kernel_optimizer.cli` run inside a
# `bash -c '... kernel_optimizer.cli ...'` matches that very shell, so an inline guard aborts
# every launch with a false positive. Here the pattern is anchored on the actual invocation
# (`python -m kernel_optimizer.cli`) and this script's own pid/ancestry is excluded.
#
# Two orchestrators sharing one GPU destroys the timings, which are the entire deliverable,
# so this check must be right rather than approximately right.
set -u

TASK="${1:?usage: launch_l3_task.sh <task> [config]}"
CFG="${2:-configs/experiments_l3_glm_linux.yaml}"
OPOP=/root/autodl-tmp/opop-workspace/opop
LOGDIR=/root/autodl-tmp/opop-workspace/opop-glm
VENV=/root/autodl-tmp/orch-venv

cd "$OPOP" || { echo "FATAL: no $OPOP"; exit 1; }

# --- guard: a real orchestrator process, not this script and not its shell ------------------
mapfile -t RUNNING < <(pgrep -af "python -m kernel_optimizer\.cli" \
                       | grep -v "launch_l3_task" | grep -v "^$$ ")
if [ "${#RUNNING[@]}" -gt 0 ]; then
    echo "ABORT: an orchestrator is already running on this box:"
    printf '  %s\n' "${RUNNING[@]}"
    exit 1
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -gt 500 ]; then
    echo "ABORT: GPU already holds ${USED} MiB; something else is using the card"
    exit 1
fi

# --- preflight: each item below has cost a real run ----------------------------------------
grep -q '"external_directory": "allow"' src/kernel_optimizer/agents/sandbox.py \
    || { echo "FATAL: external_directory is not \"allow\" -- agent calls will hang"; exit 1; }
grep -q 'OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX' "$CFG" \
    || { echo "FATAL: no output-token ceiling in $CFG -- glm-5.3 truncates on its first L3 call"; exit 1; }
"$VENV/bin/python" -c "
import sys; sys.path.insert(0, 'src')
from kernel_optimizer.gpu.worker_main import capture_timing_samples
" || { echo "FATAL: worker is missing the median fix"; exit 1; }

# The device block is written verbatim into every agent prompt, so a wrong one silently
# misinforms every agent. Compare the config's name against the card actually present.
"$VENV/bin/python" - "$CFG" <<'PY' || exit 1
import sys, torch, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
dev = (cfg.get("device") or {})
declared = str(dev.get("name", ""))
actual = torch.cuda.get_device_properties(0).name
# Compare on the model token ("RTX 4090"), since the config name carries an sm_ suffix.
key = actual.replace("NVIDIA GeForce ", "").strip()
if key not in declared:
    print(f"FATAL: config device '{declared}' does not name this card '{actual}'")
    sys.exit(1)
print(f"  ok  device block names {key}")
PY

SLUG=$(echo "$TASK" | tr ':' '-')
LOG="$LOGDIR/l3-$SLUG-$(date +%Y%m%d-%H%M%S).log"
echo "$LOG" > /root/current-run.logpath

# shellcheck disable=SC1091
source "$VENV/bin/activate"
nohup python -m kernel_optimizer.cli --config "$CFG" run --task "$TASK" > "$LOG" 2>&1 &
PID=$!
sleep 6
if ! kill -0 "$PID" 2>/dev/null; then
    echo "FATAL: orchestrator exited within 6 s; last lines of $LOG:"
    tail -20 "$LOG"
    exit 1
fi
echo "launched $TASK  pid=$PID"
echo "log=$LOG"
