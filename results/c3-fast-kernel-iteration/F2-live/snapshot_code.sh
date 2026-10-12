#!/bin/bash
# Take an IMMUTABLE snapshot of the code tree, then run the campaign against THAT copy.
#
# WHY THIS EXISTS
# ---------------
# `fast_artifacts.identify_framework` hashes ~24 source files into a `framework_id`, and every
# lane re-checks that id while it runs ("framework changed during a fixed epoch; stop rather than
# pool versions"). The guard is correct -- two candidate versions built from different code must
# not be pooled into one epoch -- but it also means ANY edit or rsync into the live tree aborts a
# run that is already in flight. That is exactly how the first F3 lane died: it was launched at
# 04:38 and a later file sync changed the hash under it.
#
# Copying the tree first decouples a multi-hour run from further development: the snapshot is
# read-only, so the working tree can keep changing while a lane is in flight.
set -euo pipefail
SRC=/root/autodl-tmp/c2-local-experiment/code/v5
DEST="$1"
mkdir -p "$(dirname "$DEST")"
mkdir -p "$DEST"
# Code and tests only. `results/` is a 179MB evidence tree reached through a symlink, and the run
# reads it from its canonical /root/autodl-tmp path regardless.
cp -r "$SRC/examples" "$SRC/src" "$SRC/tests" "$SRC/pyproject.toml" "$DEST/"
# The tests reference results/ relatively; point the snapshot at the canonical tree.
mkdir -p "$DEST/results"
ln -sfn /root/autodl-tmp/c3-fast-kernel-iteration "$DEST/results/c3-fast-kernel-iteration"
ln -sfn /root/autodl-tmp/c3-qwen3-4b-operator-goals "$DEST/results/c3-qwen3-4b-operator-goals"
# Read-only is the whole point: a snapshot that can still be edited protects nothing.
chmod -R a-w "$DEST/examples" "$DEST/src"
echo "snapshot: $DEST"
echo "c3 modules: $(find "$DEST/examples/c3_qwen3" -name '*.py' | wc -l)"
