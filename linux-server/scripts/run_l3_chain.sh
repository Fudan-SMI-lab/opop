#!/bin/bash
# Sequential L3 experiment chain for box 2. ONE run per GPU, ever.
#
# WHY A SCRIPT rather than three manual launches: timing takes an EXCLUSIVE GPU lock, so two
# concurrent runs do not merely halve throughput -- they contaminate each other's latency
# measurements, which are the entire deliverable. This script guarantees serialization by
# construction: it waits for any existing orchestrator to exit, then runs the three tasks one
# after another, each to completion.
#
# Deliberately NOT parallel and deliberately NOT backgrounded per-task. 12 h wall clock each,
# so ~36-40 h total.
set -u   # NOT -e: a failed task must not abort the remaining ones. A crash in L3:43 is a
         # reason to still collect L3:21 and L3:48, not a reason to lose them.

OPOP=/root/autodl-tmp/opop-workspace/opop
CFG=configs/experiments_l3_glm_linux.yaml
LOGDIR=/root/autodl-tmp/opop-workspace/opop-glm
CHAINLOG=$LOGDIR/l3-chain.log

exec >> "$CHAINLOG" 2>&1
echo "=============================================================="
echo "[chain] started $(date -Is)"

# --- wait for the GPU to be free -------------------------------------------------------------
# Waits on the ORCHESTRATOR, not on nvidia-smi: a run between GPU jobs (mid agent call) shows an
# idle GPU while still owning the experiment, and starting a second run then is the exact
# contamination this script exists to prevent.
while pgrep -f "kernel_optimizer.cli" > /dev/null; do
    echo "[chain] $(date -Is) waiting: an orchestrator is still running"
    sleep 120
done
echo "[chain] $(date -Is) GPU is free"

cd "$OPOP" || { echo "[chain] FATAL: cannot cd to $OPOP"; exit 1; }
# shellcheck disable=SC1091
source /root/autodl-tmp/orch-venv/bin/activate || { echo "[chain] FATAL: no orch venv"; exit 1; }

# Preflight, once, before committing 36 h. Each of these has cost a real run before:
#   external_directory   omitted -> headless ask -> 1800 s dead agent call (3 observed)
#   OUTPUT_TOKEN_MAX     unset -> glm-5.3 truncated at 32000 on its first L3 call, zero files
echo "[chain] preflight:"
grep -q '"external_directory": "allow"' src/kernel_optimizer/agents/sandbox.py \
    && echo "  ok  external_directory allow" \
    || { echo "  FATAL external_directory is not \"allow\" -- agent calls will hang 1800 s"; exit 1; }
grep -q "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX" "$CFG" \
    && echo "  ok  output token ceiling set in config" \
    || echo "  WARN output token ceiling missing -- glm-5.3 may truncate"
python -c "
import sys
sys.path.insert(0, 'src')
from kernel_optimizer.gpu.worker_main import capture_timing_samples
print('  ok  worker exposes capture_timing_samples')
" || { echo "  FATAL worker is missing the median fix"; exit 1; }

# --- the three tasks, in order --------------------------------------------------------------
# 43 first: 69.09x fusion headroom, the largest task-level lever of the three, so it is the
# strongest test of whether task_cost/bottleneck feedback actually steers the agent.
# 21 second: has a 6.92 ms / 2.18x incumbent to compare against.
# 48 last: lowest dual-precision noise floor, so the most likely to spend budget on the
# correctness gate rather than on optimization.
for TASK in level3:43 level3:21 level3:48; do
    SLUG=$(echo "$TASK" | tr ':' '-')
    LOG="$LOGDIR/l3-$SLUG-$(date +%Y%m%d-%H%M%S).log"
    echo "[chain] $(date -Is) START $TASK -> $LOG"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    python -m kernel_optimizer.cli --config "$CFG" run --task "$TASK" > "$LOG" 2>&1
    RC=$?
    echo "[chain] $(date -Is) END   $TASK rc=$RC"
    tail -3 "$LOG" | sed 's/^/[chain]   /'
    # Let the card settle and any worker subprocess reap before the next run measures ceilings.
    sleep 60
done

echo "[chain] $(date -Is) all three tasks done"
ls -dt "$LOGDIR"/runs-l3/run-*/ 2>/dev/null | head -5 | sed 's/^/[chain] run: /'
