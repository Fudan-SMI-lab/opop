#!/bin/bash
# Wrap up the step-4 C2 all-off / all-on pair. Run on box 4.
#
# WHY A SEPARATE SCRIPT rather than parameterising wrapup_s7_pair.sh: that file's --expect list, its
# pinning-evidence path and its P1-P5 reader arguments are all specific to the pair it wrapped, and it
# is now a record of what was actually run for S7. A shared script would have to be edited per pair,
# which is the same thing with a worse audit trail.
#
# WHAT DIFFERS FROM THE S7 WRAP-UP, and it changes the reading:
#   * The variable is COMPOUND -- seven C2 knobs move together -- so `audit_arm_comparability.py` is
#     given all seven and "exactly one difference" is NOT the criterion.
#   * `soft_wall` is OFF in arm A here, whereas S7 held it ON in both arms. So arm A is silent on all
#     three mechanisms and is a genuine positive control; the two pairs do not measure the same thing
#     and their numbers must not go in one table.
#   * `probe_top_k: 3` in arm B makes it non-comparable with every finished K=1 run.
#
# THE PRECONDITION is the same and is not negotiable: refuse until BOTH arms wrote RUN_FINISHED,
# because a space's "which value is best" improves monotonically with trials, so any mid-run verdict
# about an enqueued point necessarily understates it. Measured: judged by hand at 10.7 h the S7 arm
# read 4/4 enqueued points lost; after the spaces finished tuning the same table read 1 won / 3 lost /
# 2 tied, and the one that flipped had become the best value in its space.
set -uo pipefail

W=/root/autodl-tmp/work/opop
PY=/root/autodl-tmp/orch-venv/bin/python
B=/root/autodl-tmp/opop-workspace/opop-glm/runs-v3
OUT=/root/s4-wrapup.txt
# Captured DURING the run by launch_s4_c2_pair.sh, because pinning is not answerable afterwards: no
# device is recorded anywhere on disk (checked across 5689 job files on the S7 pair).
PIN_EVIDENCE=${PIN_EVIDENCE:-/root/s4-gpu-pinning.txt}

# Resolve the run dirs rather than hardcoding a timestamp: both arms are launched together and get the
# same run id, but pinning that id into the script means a relaunch silently wraps up the OLD pair.
OFF=$(ls -dt "$B"/s4-c2off/run-* 2>/dev/null | head -1)
ON=$(ls -dt "$B"/s4-c2on/run-* 2>/dev/null | head -1)
if [ -z "$OFF" ] || [ -z "$ON" ]; then
  echo "MISSING RUN DIR: off='$OFF' on='$ON' under $B -- has the pair been launched?" >&2
  exit 2
fi
if [ "$(basename "$OFF")" != "$(basename "$ON")" ]; then
  echo "WARNING: the two arms have DIFFERENT run ids:" >&2
  echo "  off: $(basename "$OFF")" >&2
  echo "  on : $(basename "$ON")" >&2
  echo "They were probably not launched together. Check before comparing anything." >&2
fi

fin() {  # 1 if the arm wrote RUN_FINISHED, else 0
  grep -c '"type": *"RUN_FINISHED"' "$1/events.jsonl" 2>/dev/null | head -1
}

a=$(fin "$OFF"); b=$(fin "$ON")
a=${a:-0}; b=${b:-0}
if [ "$a" = "0" ] || [ "$b" = "0" ]; then
  echo "NOT FINISHED YET: c2off=$a c2on=$b RUN_FINISHED event(s)."
  echo "Refusing to read an in-flight pair. Nothing was run."
  exit 4
fi

EXPECT=(--expect v3.wall_attribution.enabled
        --expect v3.wall_attribution.in_prompt
        --expect v3.wall_attribution.probe_top_k
        --expect v3.soft_wall.enabled
        --expect v3.soft_wall.in_prompt
        --expect v3.slope_guide.enabled
        --expect v3.slope_guide.use_soft_wall)

{
  echo "############ STEP-4 C2 PAIR WRAP-UP  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "arm A (C2 all-OFF, GPU 1): $OFF"
  echo "arm B (C2 all-ON,  GPU 0): $ON"
  echo
  echo "############ 0. GPU pinning (answerable only DURING the run)"
  if [ -f "$PIN_EVIDENCE" ]; then
    cat "$PIN_EVIDENCE"
  else
    echo "NO LIVE EVIDENCE FILE ($PIN_EVIDENCE) -- treat pinning as UNVERIFIED. A post-hoc run of the"
    echo "probe cannot answer it: it samples live GPU processes and nothing on disk records the device."
  fi
  echo
  echo "############ 1. INSTRUMENT SEPARATION -- is arm A really silent on all three mechanisms?"
  # The check S7 could not make: there, soft_wall ran in BOTH arms by design, so a zero in the control
  # was only expected for the sampler. Here all three must be structurally absent in arm A, and a
  # NON-zero is a config or wiring defect that invalidates the pair -- not a finding about C2.
  for label in c2off c2on; do
    d=$OFF; [ "$label" = "c2on" ] && d=$ON
    printf '%-8s' "$label"
    for t in SLOPE_GUIDE_STEP SLOPE_GUIDE_FAILED RESOURCE_SOFT_WALL RESOURCE_WALL_ATTRIBUTED \
             RESOURCE_WALL_ATTRIBUTION_FAILED; do
      n=$(grep -c "\"type\": *\"$t\"" "$d/events.jsonl" 2>/dev/null); n=${n:-0}
      printf ' %s=%s' "$t" "$n"
    done
    echo
  done
  echo "  arm A must be 0 on all five. A non-zero there means the switch did not take effect and"
  echo "  nothing below is about C2."
  echo
  echo "############ 2. search parity (align on TOTAL search, not per-space budget)"
  PYTHONPATH=$W/src $PY $W/scripts/check_arm_search_parity.py "$OFF" "$ON" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 3. config comparability -- SEVEN intended differences, not one"
  if [ -f "$W/scripts/audit_arm_comparability.py" ]; then
    PYTHONPATH=$W/src $PY $W/scripts/audit_arm_comparability.py \
      $W/configs/experiments_s4_c2off_box4gpu1.yaml \
      $W/configs/experiments_s4_c2on_box4gpu0.yaml "${EXPECT[@]}" 2>&1 || echo "(audit rc=$?)"
  else
    echo "!! AUDITOR ABSENT -- the config half is UNVERIFIED, which is not the same as comparable."
  fi
  echo
  echo "############ 4. THE RESULT (P1-P5 reader; arm B is the treatment)"
  # `--expect-2e-off control` is REQUIRED here and must NOT be passed for the S7 pair. This pair's
  # variable includes `v3.wall_attribution.enabled`, so arm A has refusals and zero wall events BY
  # DESIGN; without the flag the reader prints "2e was OFF (or crashed) ... the pair cannot be read"
  # about a perfectly healthy arm, and the response to that false alarm would be to discard 12 h of
  # work. With it, the arm is reported as the positive control it is.
  PYTHONPATH=$W/src $PY $W/scripts/analyze_s7_pair.py "$ON" "$OFF" \
    --expect-2e-off control 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 5. WHY THE MECHANISM DECLINED (counters read per-space, never summed)"
  # These counters are CUMULATIVE per SlopeGuide instance (one per tuning pass) and count two different
  # units, so summing them across events yields a triangular number. Summed by hand they read
  # `no_wall 158 / n_suggested 15` for a run whose real values were 63 and 7. This reader folds per
  # space and self-checks that sum(n_recomputes) equals the step count.
  $PY /root/probe-clean/sgc.py "$ON" "$OFF" 2>&1 || echo "(rc=$?)"
  echo
  echo "-- WHERE EACH WALL PATH ACTUALLY FAILED (three distinct causes; do NOT merge them) --"
  # `n_skipped_no_wall` is one number covering causes with completely different fixes, and reporting
  # only "zero enqueues" hides all of them. Measured at 3 h on this pair: the hard path had 3 events
  # with walls_found=0 (the refused value was inside the knob's measured range, so no wall EXISTS) and
  # 1 event where a wall DID form and the slope filter then discarded it (walls_worthless=1); the soft
  # path had 3 scans where the space's own best trial does not spill (the applicability gate) against 1
  # rejected by the monotone criterion. "Widen the wall criterion", "loosen the slope filter" and
  # "the optimum does not spill" are three different next rounds.
  $PY - "$ON" "$OFF" << 'PYEOF' 2>&1 || echo "(rc=$?)"
import json, sys
from collections import Counter
for run in sys.argv[1:]:
    rows = []
    with open(run + "/events.jsonl", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    hard = Counter()
    for r in rows:
        if r.get("type") != "RESOURCE_WALL_ATTRIBUTED":
            continue
        p = r.get("payload") or {}
        found = int(p.get("walls_found") or 0)
        worthless = int(p.get("walls_worthless") or 0)
        if found == 0:
            hard["no wall formed (refused value inside the measured range)"] += 1
        elif worthless >= found:
            hard["wall formed, slope filter discarded ALL of them"] += 1
        else:
            hard["wall formed and survived the slope filter"] += 1
    soft = Counter()
    for r in rows:
        if r.get("type") != "RESOURCE_SOFT_WALL":
            continue
        p = r.get("payload") or {}
        n = len(p.get("walls") or [])
        if not p.get("applicable"):
            soft["not applicable: %s" % str(p.get("reason"))[:56]] += 1
        elif n == 0:
            soft["applicable but 0 walls: %s" % str(p.get("reason"))[:56]] += 1
        else:
            soft["applicable, %d wall(s) produced" % n] += 1
    print("  %s" % run.rstrip("/").split("/")[-2])
    if not hard and not soft:
        print("      no wall events at all (expected in the all-OFF arm: structural zero)")
    for label, c in (("hard", hard), ("soft", soft)):
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            print("      %-4s x%-3d %s" % (label, v, k))
PYEOF
  echo
  echo "-- IS A ZERO EVEN READABLE? compare against S7 in RECOMPUTES, not hours --"
  # Hours compare the boxes; recomputes compare the mechanism. S7's treatment arm produced its FIRST
  # enqueue at recompute 41 of 72, so an arm that has done fewer recomputes than that and enqueued
  # nothing is EARLY, not barren -- at 22 recomputes mid-run this pair looked like "the all-on arm
  # produces nothing", which would have been a wrong conclusion drawn from the wrong unit.
  $PY /root/probe-clean/traj.py 2>&1 || echo "(rc=$?)"
  echo
  echo "-- DID ANY WALL TEXT REACH A REWRITER? (a primary reading of this pair) --"
  # The all-on config header names "soft walls found and delivered" and "families that received wall
  # text" as the readings that do not depend on attribution succeeding. Nothing else in this wrap-up
  # counts them, so count them here -- from the SANDBOXES, which is where delivery actually lands.
  #
  # Two traps, both already hit on the S7 pair:
  #   * hard and soft walls SHARE the filename `resource_walls.md`, so the file's existence does not say
  #     which kind arrived -- the content does. The first line is printed for each.
  #   * `analysis/` exists in BOTH analyst and rewriter sandboxes, so a `**/analysis/*` glob counts the
  #     analyst's own products as deliveries. Only `rewriter-*` sandboxes are counted.
  for label in c2on c2off; do
    d=$ON; [ "$label" = "c2off" ] && d=$OFF
    rw=$(ls -d "$d"/sandboxes/rewriter-* 2>/dev/null | wc -l)
    briefs=$(find "$d"/sandboxes/rewriter-* -name "resource_walls.md" 2>/dev/null | wc -l)
    echo "  $label: rewriter sandboxes=$rw   with resource_walls.md=$briefs"
    find "$d"/sandboxes/rewriter-* -name "resource_walls.md" 2>/dev/null | while read -r f; do
      echo "      $(basename "$(dirname "$(dirname "$f")")"): $(head -1 "$f" | cut -c1-88)"
    done
  done
  echo "  arm A's 0 is STRUCTURAL (the mechanism is off) -- the positive control, not a finding."
  echo "  Read the ratio only against the FINAL sandbox count; S7's figure was 1 of 4, and that one"
  echo "  brief was a SOFT wall, so a hard-wall claim cannot rest on it."
  echo
  echo "############ 6. DID THE ENQUEUED POINTS WIN THEIR KNOB?"
  # The reading that tests C2's premise rather than its plumbing. Pre-registered before this pair ran:
  # on S7 the steepest point (55.15% tail gain) won while shallower ones lost, and three much shallower
  # points (10.31 / 12.37 / 15.03%) then lost by only 0.3-5.5% -- i.e. slope magnitude did NOT predict
  # outcome. If that ordering appears cleanly here, the S7 reading is overturned; if it does not, C2's
  # sampler path still has exactly one positive instance.
  $PY /root/probe-clean/win.py "$ON" "$OFF" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 7. budget-matched truncation (BOTH trial counts; they can disagree)"
  $PY /root/probe-clean/bm.py "$OFF" "$ON" 2>&1 || echo "(rc=$?)"
  echo
  echo "############ 8. would a higher origin count have helped (K=3 is already set in arm B)"
  $PY /root/probe-clean/tk.py "$OFF" "$ON" 2>&1 || echo "(rc=$?)"
} > "$OUT" 2>&1

echo "wrap-up written to $OUT ($(wc -l < "$OUT") lines)"
echo
echo "############ headline lines:"
grep -E "PARITY|COMPARABLE|SEPARATED|SHARED|UNVERIFIED|AUDITOR ABSENT|enqueued|attributed|walls_found|final_reeval|same-precision|won|lost|NO ENQUEUED|HOLDS|FAILS|=> P" "$OUT" | head -40
