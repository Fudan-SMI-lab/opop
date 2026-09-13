#!/bin/bash
# Push a bundle's commits to origin/v3 from box 1, without the PAT reaching `ps` or the shell history.
#
# WHY NOT AN INLINE URL. `git push https://user:TOKEN@github.com/...` puts the token in the remote
# box's process table for the duration of the push, where any other process can read it. The token is
# already on four machines; it does not need to be in a fifth place.
#
# The token arrives on STDIN (never as an argument), is written to a 0600 file that git's `store`
# helper reads, and the file is shredded in a trap that fires on every exit path including failure.
set -euo pipefail

BUNDLE="${1:?usage: push_bundle_from_box.sh <bundle> <repo-dir>}"
REPO="${2:?usage: push_bundle_from_box.sh <bundle> <repo-dir>}"
CRED="$(mktemp)"
chmod 600 "$CRED"
cleanup() {
  shred -u "$CRED" 2>/dev/null || rm -f "$CRED"
  cd "$REPO" 2>/dev/null && git config --unset credential.helper 2>/dev/null || true
}
trap cleanup EXIT

read -r TOKEN
printf 'https://ZihangZ:%s@github.com\n' "$TOKEN" > "$CRED"
unset TOKEN

cd "$REPO"
git config credential.helper "store --file=$CRED"

# Bring the bundle's objects in under a temp ref, so nothing touches the working tree or HEAD --
# this checkout has local .linux.py variants and an experiment's history in it.
git fetch "$BUNDLE" 'refs/heads/*:refs/remotes/bundle/*' 2>&1 | tail -3
TARGET="$(git rev-parse refs/remotes/bundle/v3 2>/dev/null || git rev-parse FETCH_HEAD)"
echo "bundle head: $TARGET"

# Fast-forward only: refuse if origin/v3 has moved to something this bundle does not contain.
git push origin "$TARGET:refs/heads/v3" 2>&1 | sed 's/ghp_[A-Za-z0-9]*/[REDACTED]/g' | tail -5
echo "--- origin/v3 after push:"
git ls-remote origin v3 2>&1 | sed 's/ghp_[A-Za-z0-9]*/[REDACTED]/g'
