#!/bin/bash
# Wrap up the S7 pair, then hand off to step-4. Run on box 4.
#
# WHY A SCRIPT. The four readers must run in a fixed order and two of them need PYTHONPATH set, but
# the real reason is the PRECONDITION: every one of these tools reads the event log as if the run had
# ended. On an in-flight run they return plausible numbers about a state that no longer holds -- the
# report tool's `ended` rows are known to be untrustworthy mid-run, and the per-space budget reader
# once printed "PARITY NOT OK: 15 vs 7" on a pair that was fine, because the only space either arm
# had open WAS the in-flight one. So this refuses to run until BOTH arms have written RUN_FINISHED.
#
# ORDER, and why it is this order:
#   0. gpu_pinning_check   -- did the two arms actually use different cards? If they shared one, every
#                             latency in the pair is contaminated and nothing below is worth reading.
#   1. check_arm_search_parity -- did the wall clock buy comparable SEARCH? Align on TOTAL search, not
#                             per-space budget; a K expansion draws a whole extra budget.
#   2. audit_arm_comparability -- was the CONFIG difference the intended one only?
#   3. analyze_s7_pair     -- the actual result, read only after the three above have passed.
# Then budget_matched_arm_comparison, because on this project the two trial counts can order the arms
# OPPOSITELY (spent vs complete: 800/617 against 760/637, failures 183 vs 123).
set -uo pipefail

W=/root/autodl-tmp/work/opop
PY=/root/autodl-tmp/orch-venv/bin/python
B=/root/autodl-tmp/opop-workspace/opop-glm/runs-v3
R=run-l3-43-20260913-202332
CTL=$B/s7-control/$R
TRT=$B/s7-treatment/$R
OUT=/root/s7-wrapup.txt
# Written DURING the run by whoever launched the pair, because pinning is not answerable afterwards.
PIN_EVIDENCE=${PIN_EVIDENCE:-/root/s7-gpu-pinning.txt}

fin() {  # 1 if the arm wrote RUN_FINISHED, else 0
  grep -c '"type": *"RUN_FINISHED"' "$1/events.jsonl" 2>/dev/null | head -1
}

a=$(fin "$CTL"); b=$(fin "$TRT")
a=${a:-0}; b=${b:-0}
if [ "$a" = "0" ] || [ "$b" = "0" ]; then
  echo "NOT FINISHED YET: control=$a treatment=$b RUN_FINISHED event(s)."
  echo "Refusing to read an in-flight pair -- these tools would return plausible numbers about a"
  echo "state that no longer holds. Nothing was run."
  exit 4
fi

{
  echo "############ S7 PAIR WRAP-UP  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "############ 0. GPU pinning -- if the arms shared a card, stop reading here"
  # ⚠ THIS CANNOT BE ANSWERED HERE. The probe samples LIVE compute processes, and by wrap-up time both
  # runs have ended, so it correctly reports "NO COMPUTE PROCESSES SEEN ... ANSWERS NOTHING" (rc=1)
  # rather than reading its own silence as separation. Nothing on disk substitutes: across 5689 job
  # files in this pair, neither `jobs/*.json` nor `out.json` records a CVD, a device index or a UUID.
  # It is kept in the sequence only so the absence is VISIBLE in the report -- an unanswered
  # precondition must not look like a passed one. The answer has to be taken DURING the run:
  # `launch_s4_c2_pair.sh` now does that automatically ~7 min after launch.
  if [ -f "$PIN_EVIDENCE" ]; then
    echo "(live evidence captured during the run, from $PIN_EVIDENCE:)"
    cat "$PIN_EVIDENCE"
  else
    echo "NO LIVE EVIDENCE FILE ($PIN_EVIDENCE). Post-hoc attempt follows and is expected to answer"
    echo "nothing; treat pinning as UNVERIFIED for this pair unless it was checked while both ran."
    $PY /root/probe-clean/gpu_pinning_check.py "$CTL" "$TRT" 2>&1 || echo "(pinning check rc=$?)"
  fi
  echo
  echo "############ 1. search parity (align on TOTAL search, not per-space budget)"
  PYTHONPATH=$W/src $PY $W/scripts/check_arm_search_parity.py "$CTL" "$TRT" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 2. config comparability"
  # `[ -f ]` first: this file was MISSING on box 4 the first time this wrap-up was staged, and
  # `|| echo "(rc=$?)"` printed rc=127 -- a skipped step rendered as a completed one, which is the
  # recorded `a-skip-is-not-a-verdict` shape. An absent auditor must read as absent, not as a pass.
  if [ -f "$W/scripts/audit_arm_comparability.py" ]; then
    PYTHONPATH=$W/src $PY $W/scripts/audit_arm_comparability.py \
      $W/configs/experiments_s7_control_box4gpu1.yaml \
      $W/configs/experiments_s7_treatment_box4gpu0.yaml \
      --expect v3.slope_guide.enabled 2>&1 || echo "(audit rc=$?)"
  else
    echo "!! AUDITOR ABSENT: $W/scripts/audit_arm_comparability.py does not exist."
    echo "   The config half of this pair is UNVERIFIED -- that is not the same as comparable."
  fi
  echo
  echo "############ 3. THE RESULT"
  PYTHONPATH=$W/src $PY $W/scripts/analyze_s7_pair.py "$TRT" "$CTL" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 4. budget-matched truncation (BOTH trial counts, they can disagree)"
  $PY /root/probe-clean/bm.py "$CTL" "$TRT" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 5. would a higher origin count have helped this pair"
  $PY /root/probe-clean/tk.py "$CTL" "$TRT" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 6. DID THE ENQUEUED POINTS WIN THEIR KNOB?"
  # The reading that turns S7's result from a claim into a measurement. `analyze_s7_pair.py` reports
  # the enqueue count and that the points were drawn, and those two facts read as a success -- but
  # "was measured" is not "was good", and C2's premise is specifically that a steep truncated
  # direction is worth measuring. Runs LAST and only after the RUN_FINISHED gate above, because a
  # space's "which value is best" improves monotonically with trials: judged mid-run it necessarily
  # understates the enqueued value, and doing exactly that by hand at 10.7 h produced a "4/4 lost"
  # verdict that became 1 won / 2 lost / 1 too-thin once the space finished tuning.
  $PY /root/probe-clean/win.py "$TRT" "$CTL" 2>&1 || echo "(rc=$?)"
} > "$OUT" 2>&1

echo "wrap-up written to $OUT ($(wc -l < "$OUT") lines)"
echo
echo "############ headline lines:"
grep -E "PARITY|COMPARABLE|SHARED|pinned|separate|enqueued|attributed|walls_found|final_reeval|NOT |won|lost|NO ENQUEUED|AUDITOR ABSENT" "$OUT" | head -40
