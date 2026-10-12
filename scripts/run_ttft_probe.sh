#!/bin/bash
# One ttft dev->freeze->lane chain at ONE site, on ONE card, isolated so a sibling can run the same
# on the other card at a different site. This is how "rerun ttft with the improved acceptance rule"
# and "search a second ttft site in parallel" become two invocations of the SAME tested path.
#
# WHY A NEW LAUNCHER RATHER THAN run_chain.sh
# -------------------------------------------
# run_chain.sh hardcodes CUDA_VISIBLE_DEVICES=0, one agent port, one runtime root, and runs the full
# 36-cell matrix. Two lanes on two cards need the per-process isolation that run_f3.py and
# run_formal_campaign.py already accept (C3_CUDA_INDEX / C3_AGENT_PORT / C3_RUNTIME_ROOT) and need the
# matrix skipped (C3_SKIP_MATRIX=1, C3_LANES=ttft) so each process runs ONLY its ttft lane. This
# wrapper sets exactly those and otherwise calls the identical run_f3.py + run_formal_campaign.py, so
# no scientific code is duplicated -- only the tested primitives are wired for two-card isolation.
#
# The improved ACCEPTANCE rule (noise-aware, best-incumbent, patience) rides inside those primitives:
# it is part of framework_id (fast_acceptance.py + the edited fast_workflow.py/runner_measure.py), so
# both the dev epoch and the lane run under it automatically. Nothing here selects it.
#
# Usage:
#   run_ttft_probe.sh <site_module> <card_index> <stamp_suffix> [dev_hours] [search_hours]
set -euo pipefail
SITE="$1"                      # e.g. model.norm  or  model.layers.0.mlp.act_fn
CARD="$2"                      # 0 or 1
SUFFIX="$3"                    # short tag for the run dirs, e.g. norm / silu
DEV_HOURS="${4:-4}"
SEARCH_HOURS="${5:-3}"

PY=/root/autodl-tmp/kernel-opt-venv/bin/python
STAMP=$(date +%m%d-%H%M%S)
ROOT=/root/autodl-tmp/c3-fast-kernel-iteration
SNAP=/root/autodl-tmp/c3-snapshots/probe-$SUFFIX-$STAMP
OUT_DEV=$ROOT/F5-probe-$SUFFIX-dev-$STAMP
OUT_CAMP=$ROOT/F5-probe-$SUFFIX-lane-$STAMP
CFG=$ROOT/development-996318d/agent-config.json
REV="probe-$SUFFIX-$STAMP"

# An IMMUTABLE snapshot: framework_id is re-checked while a lane runs, so a later edit or sync into
# the live tree would abort it. Each probe gets its OWN snapshot so the two chains never share bytes.
/root/autodl-tmp/snapshot_code.sh "$SNAP"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PYTHONPATH="$SNAP/src:$SNAP"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# The site under test, threaded into BOTH the dev epoch and the campaign so they cannot disagree.
export C3_SITE_MODULE="$SITE"
export C3_ALLOW_COMPOSITE=""

# PER-CARD ISOLATION. Distinct card, agent port, and runtime root (triton/torch/XDG caches) so two
# chains on one box never race the GPU, the opencode port, or a compile cache.
export C3_CUDA_INDEX="$CARD"
export C3_AGENT_PORT="$((4210 + CARD))"
export C3_RUNTIME_ROOT="/root/autodl-tmp/probe-runtime-$SUFFIX"
mkdir -p "$C3_RUNTIME_ROOT"

# Only the ttft lane, and no matrix: this process produces one ttft row and its measured gain, which
# is what the site-space ranking reads. A matrix here would fill single/multi with baseline fallbacks.
export C3_LANES="ttft"
export C3_SKIP_MATRIX="1"

cd "$SNAP"
echo "{\"event\":\"PROBE_START\",\"site\":\"$SITE\",\"card\":\"$CARD\",\"port\":\"$C3_AGENT_PORT\",\"snapshot\":\"$SNAP\",\"dev\":\"$OUT_DEV\",\"camp\":\"$OUT_CAMP\"}"

# Development at the ttft site until DEV_GO (seals the improved-acceptance framework_id).
$PY /root/autodl-tmp/run_f3.py "$OUT_DEV" "$SNAP" "$REV" "$CFG" "$DEV_HOURS"

STATUS=$($PY -c "
import json,glob
p=sorted(glob.glob('$OUT_DEV/F*/result.json'))
print(json.load(open(p[0]))['status'] if p else 'NO_RESULT')
")
echo "{\"event\":\"PROBE_DEV_STATUS\",\"site\":\"$SITE\",\"status\":\"$STATUS\"}"
if [ "$STATUS" != "DEV_GO" ]; then
  echo "{\"event\":\"PROBE_STOP\",\"reason\":\"development did not reach DEV_GO\",\"status\":\"$STATUS\"}"
  exit 1
fi

# Freeze (all three goals at this forced site) then run ONLY the ttft lane. `-` is the optional C2
# receipt (absent: C2 isolation is structural). FINAL_HOURS=1 is unused because the matrix is skipped.
$PY /root/autodl-tmp/run_formal_campaign.py "$OUT_DEV" "$OUT_CAMP" "$SNAP" - "$SEARCH_HOURS" "1"

echo "{\"event\":\"PROBE_DONE\",\"site\":\"$SITE\",\"card\":\"$CARD\",\"camp\":\"$OUT_CAMP\"}"
