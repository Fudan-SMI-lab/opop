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
  # Delete the temp refs. A LEFTOVER refs/remotes/bundle/* is not inert: a stale one from an earlier
  # push resolved to an unrelated old commit that git then tried to push, which is how this script
  # once nearly published a months-old tree. Cleaning up is part of the operation, not tidiness.
  cd "$REPO" 2>/dev/null && git for-each-ref --format='%(refname)' refs/remotes/bundle \
    | while read -r r; do git update-ref -d "$r"; done 2>/dev/null || true
}
trap cleanup EXIT

read -r TOKEN
printf 'https://ZihangZ:%s@github.com\n' "$TOKEN" > "$CRED"
unset TOKEN

cd "$REPO"
git config credential.helper "store --file=$CRED"

# Bring the bundle's objects in under a temp ref, so nothing touches the working tree or HEAD --
# this checkout has local .linux.py variants and an experiment's history in it.
#
# WHY THE HEAD FALLBACK. A bundle created from a commit RANGE (`git bundle create f A..B`) contains
# only the ref `HEAD`, not `refs/heads/<branch>`. Fetching 'refs/heads/*:refs/remotes/bundle/*' then
# matches NOTHING -- and because nothing matched, git does not write FETCH_HEAD either, so the
# `|| git rev-parse FETCH_HEAD` fallback died with "ambiguous argument 'FETCH_HEAD'" rather than
# falling through. Ask the bundle what it actually contains instead of assuming a shape.
HAVE_BRANCH=$(git bundle list-heads "$BUNDLE" | grep -c 'refs/heads/' || true)
if [ "$HAVE_BRANCH" -gt 0 ]; then
  git fetch "$BUNDLE" 'refs/heads/*:refs/remotes/bundle/*' 2>&1 | tail -3
  TARGET="$(git rev-parse refs/remotes/bundle/v3 2>/dev/null || git rev-parse FETCH_HEAD)"
else
  git fetch "$BUNDLE" 'HEAD:refs/remotes/bundle/head' 2>&1 | tail -3
  TARGET="$(git rev-parse refs/remotes/bundle/head)"
fi
echo "bundle head: $TARGET"

# Fast-forward only: refuse if origin/v3 has moved to something this bundle does not contain.
git push origin "$TARGET:refs/heads/v3" 2>&1 | sed 's/ghp_[A-Za-z0-9]*/[REDACTED]/g' | tail -5
echo "--- origin/v3 after push:"
git ls-remote origin v3 2>&1 | sed 's/ghp_[A-Za-z0-9]*/[REDACTED]/g'
