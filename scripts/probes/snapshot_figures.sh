#!/bin/bash
# Current figures for the S7 pair snapshot. A FILE, not an inline ssh command with nested quotes --
# that has now failed three times in this session against my own recorded rule.
set -u
B=/root/autodl-tmp/opop-workspace/opop-glm/runs-v3
R=run-l3-43-20260913-202332
PY=/root/autodl-tmp/orch-venv/bin/python
export PYTHONPATH=/root/autodl-tmp/work/opop/src

echo "===== PARITY"
$PY /root/probe-clean/check_arm_search_parity.py "$B/s7-control/$R" "$B/s7-treatment/$R" 2>&1 \
  | grep -E "span |rate |MEDIAN|TOTAL SEARCH|PARITY|per-space budget|rewrite rounds"

echo
echo "===== SPACES"
$PY /root/probe-clean/space_accounting.py 2>&1 | grep -E "^===|spaces |REWRITE_PRODUCED|SPACE_EXPANDED"

echo
echo "===== REWRITE vs PROMISE"
$PY /root/probe-clean/rewrite_vs_promise.py 2>&1 | grep -E "^===|^  ---|parent |child |NO VERDICT|resource dims"
