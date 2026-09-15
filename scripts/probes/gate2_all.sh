#!/bin/bash
# Gate 2, one command per finished run. Order matters: the gate-2 readout REFUSES an
# unfinished run (verdicts are unreadable mid-run), so it runs first and its exit code
# decides whether the rest is meaningful at all.
#
# WHY A WRAPPER: gate 2 is budgeted at 1-2 h and four probes now feed it. Typing them by
# hand at that moment is how a step gets skipped -- and the one most likely to be skipped
# is the uptake tier, which is the only one that can support the mechanism claim.
set -uo pipefail
W=${OPOP_WORK:-/root/autodl-tmp/work/opop}
PY=${OPOP_PY:-/root/autodl-tmp/orch-venv/bin/python}
PROBE=${OPOP_PROBE_DIR:-/root/probe-clean}
for run in "$@"; do
  run=${run%/}
  echo "################################################################"
  echo "# $(basename "$(dirname "$run")")  $(basename "$run")"
  echo "################################################################"
  echo "--- [1/4] gate-2 funnel (refuses an unfinished run) ---"
  if ! PYTHONPATH="$W/src" "$PY" "$PROBE/v41_gate2_readout.py" "$run"; then
    echo "!! gate-2 readout refused or failed for $run -- SKIPPING the rest of its table,"
    echo "!! because every number below is a mid-run reading if the run has not finished."
    continue
  fi
  echo
  echo "--- [2/4] P2' (primary endpoint: conditioned vs marginal, same cutoff) ---"
  PYTHONPATH="$W/src" "$PY" "$PROBE/v41_p2prime.py" "$run"
  echo
  echo "--- [3/4] rewrite provenance (tiers 1-3: what the brief made available) ---"
  "$PY" "$PROBE/rewrite_provenance.py" "$run" | grep -E \
    'families with a CONDITIONED BRIEF|candidate_id |GATE-PASSING|NONE PASSED|a brief reached|CANNOT have been'
  echo
  echo "--- [4/4] axis uptake (tier 4: the only tier that supports 'slope steered it') ---"
  "$PY" "$PROBE/v41_axis_uptake.py" "$run"
done

# The A/A pair is read as a PAIR, not per arm: its whole output is a difference. Pass both
# arms as OPOP_N1_A / OPOP_N1_B to get the paired noise floor that prices P3.
if [ -n "${OPOP_N1_A:-}" ] && [ -n "${OPOP_N1_B:-}" ]; then
  echo
  echo "################################################################"
  echo "# N1 paired noise floor (the P3 denominator)"
  echo "################################################################"
  "$PY" "$PROBE/v41_n1_noise_floor.py" "$OPOP_N1_A" "$OPOP_N1_B"
fi
