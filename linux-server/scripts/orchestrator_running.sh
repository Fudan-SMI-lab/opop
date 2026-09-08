#!/usr/bin/env bash
# Is an orchestrator ACTUALLY running on this box? Exit 0 = yes, 1 = no.
#
# The whole point is not matching itself. `pgrep -f kernel_optimizer.cli` inside any wrapper
# whose own command line contains that string matches the wrapper -- so a monitor loop built
# around that check reports "still running" forever, and two such loops keep each other alive.
# Measured: box 2's L1 smoke wrote RUN_FINISHED at 01:12:37 and two monitor loops were still
# reporting it as running 36 minutes later, each matching the other's command line. The
# completion notification never arrived, because the condition it waited on could never become
# false.
#
# Fixes it three ways at once:
#   * match on `python -m kernel_optimizer.cli`, which a shell wrapper does not contain verbatim
#     unless it is launching one;
#   * exclude this script's own pid and its parent;
#   * exclude anything whose command is a shell (`bash -c`, `sh -c`), since the orchestrator is
#     never invoked that way by the launcher.
#
# Usage:
#   orchestrator_running.sh              # exit status only
#   orchestrator_running.sh -v           # also print the matching line(s)
#
# In a wait loop, ALWAYS use this rather than an inline pgrep:
#   until ! /root/orchestrator_running.sh; do sleep 60; done; echo IDLE
set -u

VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

SELF=$$
PARENT=$PPID

MATCHES=$(pgrep -af 'python -m kernel_optimizer\.cli' 2>/dev/null \
          | grep -v -E '(^|[^0-9])'"$SELF"' ' \
          | grep -v -E '(^|[^0-9])'"$PARENT"' ' \
          | grep -v -E ' (bash|sh|dash) -c ' \
          | grep -v 'orchestrator_running' || true)

if [ -n "$MATCHES" ]; then
    [ "$VERBOSE" = 1 ] && echo "$MATCHES"
    exit 0
fi
[ "$VERBOSE" = 1 ] && echo "no orchestrator running"
exit 1
