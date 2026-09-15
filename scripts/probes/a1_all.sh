#!/bin/bash
# A1: the closing analysis of window 2, in the pre-registered order.
#
# WHY A WRAPPER, AND WHY IT REFUSES FIRST. A1 is read ONCE, after both pairs finish, and its
# output is the paper's evidence chain. Two failure modes have already bitten this project:
#
#   1. Reading a run mid-flight. "Who is best in this space" improves monotonically with
#      trials, so a mid-run verdict can be the OPPOSITE of the final one -- recorded in
#      `judging-an-enqueued-point-mid-run-reversed-the-verdict`. So step 0 REFUSES any run
#      without RUN_FINISHED, and refuses the whole pair if either arm is unfinished. A
#      partial A1 is worse than no A1: it looks like a result.
#   2. Typing the probes by hand at the moment they are needed, and skipping the one that
#      carries the mechanism claim. That is why gate2_all.sh exists; this is its A1 analogue.
#
# PAIRS, NOT RUNS. Every number here is a within-pair contrast. box4 (8352V) and box1 (8358P)
# differ 32% in compile time, and m2b runs a DIFFERENT TASK, so the two pairs are reported
# side by side and NEVER pooled. The script enforces this by taking one pair at a time.
#
# NO GPU. Reads events.jsonl and the on-disk artifacts. No torch, no CUDA, no candidate
# execution. Safe to run while anything else is running.
set -uo pipefail

W=${OPOP_WORK:-/root/autodl-tmp/work/opop}
PY=${OPOP_PY:-/root/autodl-tmp/orch-venv/bin/python}
PROBE=${OPOP_PROBE_DIR:-/root/probe-clean}
export PYTHONPATH="$W/src"

if [ $# -ne 2 ]; then
  cat >&2 <<USAGE
usage: $0 <off-arm-run-dir> <active-arm-run-dir>

  Both must be COMPLETE runs of the SAME pair on the SAME box.
  Order matters: the off (control) arm first, the active (treatment) arm second --
  every arm-difference below is read as active minus off.

  e.g. $0 .../runs-v4/m1-a/run-l3-43-... .../runs-v4/m1-b/run-l3-43-...
USAGE
  exit 2
fi
OFF=${1%/}
ACT=${2%/}

echo "################################################################################"
echo "# A1  off=$(basename "$(dirname "$OFF")")  active=$(basename "$(dirname "$ACT")")"
echo "################################################################################"

# ---- step 0: refuse unfinished runs, and refuse a broken pair ------------------------
echo
echo "=== [0/7] completeness gate (a mid-run read is not a result) ==="
bad=0
for r in "$OFF" "$ACT"; do
  if ! grep -q '"RUN_FINISHED"' "$r/events.jsonl" 2>/dev/null; then
    echo "!! NOT FINISHED: $r"
    bad=1
  else
    echo "   finished: $(basename "$(dirname "$r")")"
  fi
done
if [ "$bad" -ne 0 ]; then
  echo "!! REFUSING to read this pair. Every verdict below would be a mid-run reading, and"
  echo "!! a space's best improves monotonically with trials, so the sign can still flip."
  exit 1
fi

# ---- step 1: comparability, BEFORE any effect size ----------------------------------
# An effect read across arms that did not buy comparable search is not an effect. Wall
# clock is the binding budget on every finished run so far, so equal trial COUNTS are not
# the check -- GPU wall is.
echo
echo "=== [1/7] arm comparability: GPU wall, trials, failures, families ==="
"$PY" "$PROBE/a1_pair_comparability.py" "$OFF" "$ACT"

# ---- step 2: A1-2 dose separation ---------------------------------------------------
# Does the independent variable actually differ? Window 1's n1 pair had mode=off on BOTH
# arms, so it could not test C2 at all. Read this before believing any contrast.
echo
echo "=== [2/7] A1-2 dose separation: did the mechanism only fire in the active arm? ==="
"$PY" "$PROBE/a1_pair_comparability.py" "$OFF" "$ACT" --mechanism

# ---- step 3: A1-1 P2', the primary scientific endpoint ------------------------------
# Five definitions side by side so no single number can hide which defect moved it.
# `noleak` is the reportable one; old/signfix/scoped are retracted diagnostics.
echo
echo "=== [3/7] A1-1 P2' PRIMARY ENDPOINT -- active arm (off arm has no contrasts) ==="
"$PY" "$PROBE/v41_p2prime.py" "$ACT"
echo
echo "--- off arm, for the record (expected: no contrast; that is by design, not a result) ---"
"$PY" "$PROBE/v41_p2prime.py" "$OFF" 2>&1 | tail -5

# ---- step 4: A1-4 brief attribution, tiers 1-5 -------------------------------------
# Delivery count > 0 is NOT the mechanism working: window 1 delivered 4 briefs while both
# rewrites happened in families that received none. Tiers must be read in order.
echo
echo "=== [4/7] A1-4 brief attribution ladder ==="
echo "--- tiers 1-3 (delivered / parent briefed / parent had a gate-passing slope) ---"
"$PY" "$PROBE/rewrite_provenance.py" "$ACT" | grep -E \
  'families with a CONDITIONED BRIEF|candidate_id |GATE-PASSING|NONE PASSED|a brief reached|CANNOT have been'
echo
echo "--- tier 4 (did the rewrite MOVE the briefed axis) ---"
"$PY" "$PROBE/v41_axis_uptake.py" "$ACT"
echo
echo "--- tier 5 (did the refusal rate MEASURABLY fall) -- BOTH arms, the arm DIFFERENCE is the reading ---"
echo "    off arm is the control: window 1 saw 4/8 families relieve a wall with the mechanism OFF."
for r in "$OFF" "$ACT"; do
  echo "  [$(basename "$(dirname "$r")")]"
  "$PY" "$PROBE/v41_refusal_delta.py" "$r"
done

# ---- step 5: A1-3 enqueued-point outcomes ------------------------------------------
echo
echo "=== [5/7] A1-3 did the enqueued values WIN their space? (post-run only) ==="
"$PY" "$PROBE/did_the_enqueued_point_win.py" "$ACT"

# ---- step 6: A1-5 end-to-end, reported with its power limit -------------------------
# NOT a primary criterion. Window 2 has no qualified tie-band calibration (see
# docs/result-window1-offline-closeout.md section 5), so this is descriptive.
echo
echo "=== [6/7] A1-5 end-to-end latency (DESCRIPTIVE: no qualified tie band exists) ==="
"$PY" "$PROBE/a1_pair_comparability.py" "$OFF" "$ACT" --final

# ---- step 7: instrument health ------------------------------------------------------
echo
echo "=== [7/7] gate-2 funnel on the active arm (instrument health, not an endpoint) ==="
"$PY" "$PROBE/v41_gate2_readout.py" "$ACT"

echo
echo "################################################################################"
echo "# A1 done for this pair. Reminders before any number leaves this output:"
echo "#   - do NOT pool the two pairs (different CPU; m2b is a different task)"
echo "#   - final values come from RUN_FINISHED.summary.best.final_reeval_median_ms"
echo "#   - speedup_vs_eager is cross-precision and is NOT reportable"
echo "#   - report P2' multiples WITH their definition and n"
echo "################################################################################"
