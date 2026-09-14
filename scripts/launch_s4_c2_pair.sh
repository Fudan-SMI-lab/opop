#!/bin/bash
# Launch the step-4 C2 all-off / all-on pair on box 4, one arm per GPU, simultaneously.
#
# WHY A SCRIPT. Four things must be right at once and each has bitten this project before:
#   * CUDA_VISIBLE_DEVICES is set in the LAUNCH COMMAND, not in the config -- a config inspection
#     cannot tell you which card an arm used, and one nvidia-smi snapshot cannot prove separation
#     (jobs are one-shot subprocesses, so the gaps read 0% / 0 MiB either way).
#   * XDG_DATA_HOME must differ per arm or the two opencode servers share state.
#   * PYTHONPATH=src, because the orchestrator venv does not have the package installed.
#   * the two arms must start together, or the later one runs against a differently-loaded box.
#
# PRECONDITION, checked below rather than trusted: no other orchestrator is running. Starting this
# pair while the S7 pair is still going would put four runs on two cards and every timing would be
# contaminated.
set -uo pipefail

W=/root/autodl-tmp/work/opop
PY=/root/autodl-tmp/orch-venv/bin/python
TASK=level3:43

cd "$W" || { echo "cannot cd to $W" >&2; exit 2; }

# `[k]ernel_optimizer` so this grep does not match its own command line. VERIFIED, not assumed: run
# bare it reports 2 against a ground truth of 2. It reports 3 when the INVOKING command also carries
# the string unbracketed (an `echo` label, say) -- the bracket protects only the occurrence it is in,
# so a wrapper that names the pattern inflates the count. Since this number decides whether it is safe
# to start, and a self-match can only inflate, that direction is the safe one; do not "fix" it by
# loosening the pattern.
running=$(ps -eo args | grep -c "[k]ernel_optimizer.cli" || true)
if [ "$running" != "0" ]; then
  echo "REFUSING TO START: $running orchestrator process(es) already running." >&2
  echo "The step-4 pair needs both cards to itself. Wait for the current pair to finish." >&2
  ps -eo pid,etimes,args | grep "[k]ernel_optimizer.cli" >&2
  exit 3
fi

for f in configs/experiments_s4_c2off_box4gpu1.yaml configs/experiments_s4_c2on_box4gpu0.yaml; do
  [ -f "$f" ] || { echo "missing config: $f" >&2; exit 2; }
done

# COMPARABILITY, checked BEFORE spending 12 h rather than discovered in the wrap-up. This pair is a
# deliberate COMPOUND variable -- seven C2 knobs move together -- so the audit cannot be "exactly one
# difference"; it has to be "exactly these seven and nothing else". The seven are listed here rather
# than left to whoever runs the audit later: an --expect list carried in someone's head is how the
# first audit of this pair reported a false defect (`ordered_categoricals`, which the config
# deliberately leaves off in BOTH arms because it is a SAMPLER change, not a wall-mechanism one, and
# two sampling changes at once are unattributable).
EXPECT=(--expect v3.wall_attribution.enabled
        --expect v3.wall_attribution.in_prompt
        --expect v3.wall_attribution.probe_top_k
        --expect v3.soft_wall.enabled
        --expect v3.soft_wall.in_prompt
        --expect v3.slope_guide.enabled
        --expect v3.slope_guide.use_soft_wall)
if ! PYTHONPATH="$W/src" "$PY" scripts/audit_arm_comparability.py \
      configs/experiments_s4_c2off_box4gpu1.yaml \
      configs/experiments_s4_c2on_box4gpu0.yaml "${EXPECT[@]}" > /root/s4-preaudit.txt 2>&1; then
  echo "REFUSING TO START: config audit failed. See /root/s4-preaudit.txt" >&2
  tail -20 /root/s4-preaudit.txt >&2
  exit 5
fi
echo "config audit: $(grep -c 'THE VARIABLE' /root/s4-preaudit.txt) intended difference(s), COMPARABLE"

mkdir -p /root/autodl-tmp/opop-workspace/opop-glm/runs-v3/s4-c2off \
         /root/autodl-tmp/opop-workspace/opop-glm/runs-v3/s4-c2on

echo "starting arm A (C2 all-OFF) on GPU 1"
CUDA_VISIBLE_DEVICES=1 XDG_DATA_HOME=/root/autodl-tmp/xdg-s4a PYTHONPATH=src \
  nohup "$PY" -m kernel_optimizer.cli \
  --config configs/experiments_s4_c2off_box4gpu1.yaml run --task "$TASK" \
  > /root/s4-c2off.log 2>&1 &
A=$!

echo "starting arm B (C2 all-ON) on GPU 0"
CUDA_VISIBLE_DEVICES=0 XDG_DATA_HOME=/root/autodl-tmp/xdg-s4b PYTHONPATH=src \
  nohup "$PY" -m kernel_optimizer.cli \
  --config configs/experiments_s4_c2on_box4gpu0.yaml run --task "$TASK" \
  > /root/s4-c2on.log 2>&1 &
B=$!

echo "arm A pid=$A (GPU 1, C2 off)   arm B pid=$B (GPU 0, C2 on)"
echo "logs: /root/s4-c2off.log  /root/s4-c2on.log"
sleep 20
for p in "$A" "$B"; do
  if ! kill -0 "$p" 2>/dev/null; then
    echo "WARNING: pid $p died within 20 s -- read its log before assuming the pair is running" >&2
  fi
done
ps -eo pid,etimes,args | grep "[k]ernel_optimizer.cli"
