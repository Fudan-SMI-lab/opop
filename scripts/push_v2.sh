#!/bin/bash
# Push v2 to GitHub, and only report success when the REMOTE HASH MATCHES LOCAL.
#
# Two failure modes this exists to catch:
#   1. A real TLS failure can still print "Everything up-to-date" (or a normal-looking
#      "old..new HEAD -> v2"), so push output is never proof. Only ls-remote is.
#   2. If ls-remote itself cannot run (network cut, command killed, timeout), that is
#      UNKNOWN, not success -- it must exit nonzero and say so, never be read as verified.
#
# Usage:  GH_PAT=<token> bash scripts/push_v2.sh
# The token is used only in a one-time inline URL; it is never written to git config,
# remotes, or .git-credentials.
set -u
cd /mnt/d/Pyhon_projects/opop/v2 || exit 1

if [ -z "${GH_PAT:-}" ]; then
  echo "PUSH_ABORTED: GH_PAT is not set" >&2
  exit 2
fi
URL="https://ZihangZ:${GH_PAT}@github.com/Fudan-SMI-lab/opop.git"
redact() { sed -e "s/${GH_PAT}/<redacted>/g"; }

L=$(git rev-parse HEAD) || exit 1
echo "local HEAD = $L"

for i in 1 2 3; do
  git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 \
      push "$URL" HEAD:v2 >/tmp/push.txt 2>&1
  push_rc=$?

  # Verification is a separate, independently-checked step. Its own exit code matters:
  # an empty $R could mean "branch absent" OR "ls-remote never ran".
  R=$(git ls-remote "$URL" refs/heads/v2 2>/tmp/lsr.txt | cut -f1)
  lsr_rc=$?

  if [ "$lsr_rc" -ne 0 ]; then
    echo "attempt $i: push_rc=$push_rc but VERIFICATION FAILED TO RUN (ls-remote rc=$lsr_rc)"
    redact </tmp/lsr.txt | tail -2
    echo "PUSH_UNVERIFIED: remote state unknown -- do NOT report this as pushed" >&2
    exit 3
  fi

  echo "attempt $i: push_rc=$push_rc remote=${R:-<none>} local=$L"
  if [ "$R" = "$L" ]; then
    echo "PUSH_OK: remote matches local at $L"
    exit 0
  fi
  redact </tmp/push.txt | tail -3
  sleep 5
done

echo "PUSH_FAILED: remote never matched local $L" >&2
exit 1
