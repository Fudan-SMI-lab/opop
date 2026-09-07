#!/bin/bash
# Does the chain's wait predicate reliably see a live orchestrator?
#
# Two contradictory readings appeared within one minute:
#   - `pgrep -f "kernel_optimizer.cli" | wc -l` -> 0   (would let the chain START)
#   - the chain's own loop, same predicate       -> waiting (correct)
# while `ps -eo cmd | grep` showed exactly one live orchestrator, PID 54470.
#
# If the 0 is real and intermittent, the chain can launch L3 on top of a running experiment and
# contaminate both sets of timings -- the one failure this script exists to prevent. So resolve
# it before 36 h of GPU time depends on it. Run it 20 times and report EVERY disagreement.
echo "reference (ps): $(ps -eo pid,cmd | grep -c '[k]ernel_optimizer.cli') orchestrator(s)"
echo
disagree=0
for i in $(seq 1 20); do
    if pgrep -f "kernel_optimizer.cli" > /dev/null; then p=1; else p=0; fi
    r=$(ps -eo pid,cmd | grep -c '[k]ernel_optimizer.cli')
    if [ "$r" -gt 0 ]; then expect=1; else expect=0; fi
    if [ "$p" != "$expect" ]; then
        echo "  DISAGREE run $i: pgrep says $p, ps says $r processes"
        disagree=$((disagree + 1))
    fi
    sleep 0.3
done
echo
if [ "$disagree" -gt 0 ]; then
    echo "FAIL: $disagree of 20 disagreed -- the wait predicate is NOT reliable"
    exit 1
fi
echo "PASS: 20 of 20 agreed with ps"
# Positive control: the predicate must also be able to say NO, or agreement proves nothing.
if pgrep -f "this-pattern-matches-no-process-at-all-xyzzy" > /dev/null; then
    echo "CONTROL FAILED: pgrep matched a nonexistent pattern; it cannot report absence"
    exit 1
fi
echo "control ok: pgrep correctly reports absence for a pattern nothing matches"
