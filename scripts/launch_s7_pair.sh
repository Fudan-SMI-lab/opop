#!/bin/bash
# Launch the S7 pair on box 4: treatment on GPU 0, control on GPU 1, simultaneously.
#
# WHY A SCRIPT RATHER THAN TWO INLINE SSH COMMANDS. Two reasons, both learned here:
#   * an inline `nohup ... &` over ssh can be killed when the ssh channel closes, and the orchestrator
#     has NO signal handler -- SIGTERM skips `OpencodeServer.stop()` and leaves an `opencode serve`
#     orphan holding the port. setsid detaches it from the session so channel teardown cannot reach it.
#   * `opencode` is only on the LOGIN shell's PATH, so the orchestrator must be started from a login
#     shell or every agent call fails with "opencode not found" after the GPU work has already run.
set -u

REPO=/root/autodl-tmp/work/opop
PY=/root/autodl-tmp/orch-venv/bin/python
TASK=level3:43
cd "$REPO" || exit 1

# Fail loudly BEFORE spending 24 GPU-hours if the checkout is not the commit the pair was designed
# against. A pair run from two different code states is not a pair.
WANT=39f01ea
HAVE=$(git rev-parse --short HEAD)
if [ "$HAVE" != "$WANT" ]; then
  echo "REFUSING: HEAD is $HAVE, expected $WANT" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING: working tree is dirty; both arms must run the same bytes" >&2
  git status --porcelain >&2
  exit 1
fi

launch() {
  local name=$1 gpu=$2 xdg=$3 cfg=$4
  local log=/root/autodl-tmp/s7-$name.log
  # One arm per card. CUDA_VISIBLE_DEVICES pins it; XDG_DATA_HOME keeps the two opencode SQLite states
  # apart (one db written by two processes otherwise).
  setsid env CUDA_VISIBLE_DEVICES="$gpu" XDG_DATA_HOME="$xdg" PYTHONPATH=src PYTHONIOENCODING=utf-8 \
    bash -lc "'$PY' -m kernel_optimizer.cli --config '$cfg' run --task $TASK" \
    > "$log" 2>&1 < /dev/null &
  echo "$name: pid $! gpu $gpu log $log"
}

launch treatment 0 /root/autodl-tmp/xdg-s7t configs/experiments_s7_treatment_box4gpu0.yaml
launch control   1 /root/autodl-tmp/xdg-s7c configs/experiments_s7_control_box4gpu1.yaml

sleep 20
echo "--- after 20s:"
pgrep -af "[k]ernel_optimizer" | sed 's/^/  /'
echo "--- log heads:"
for n in treatment control; do
  echo "  == $n:"; tail -3 "/root/autodl-tmp/s7-$n.log" 2>/dev/null | sed 's/^/     /'
done
