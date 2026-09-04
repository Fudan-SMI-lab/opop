#!/bin/bash
# Final A/B of worker startup, using the venv the harness is now CONFIGURED to use
# (~/kernel-opt-venv on ext4) against the old shared one on 9p. Alternates so neither
# side benefits from page-cache warmth. Each iteration imports torch/triton/kernelbench
# and compiles+launches a real triton kernel, verified with allclose.
B=/mnt/d/Pyhon_projects/opop/v2/scripts/bench_worker_startup.py
NEW=$HOME/kernel-opt-venv/bin/python
OLD=/mnt/d/Pyhon_projects/opop/kernelfoundry/.venv-wsl/bin/python
CNEW=$HOME/.triton-cache-kopt
COLD=/mnt/d/Pyhon_projects/opop/v2/.triton-cache-wsl
mkdir -p "$CNEW"

TRITON_CACHE_DIR=$CNEW "$NEW" "$B" >/dev/null 2>&1
TRITON_CACHE_DIR=$COLD "$OLD" "$B" >/dev/null 2>&1

for i in 1 2 3; do
  echo -n "ext4 venv (configured) run$i: "
  TRITON_CACHE_DIR=$CNEW "$NEW" "$B"
  echo -n "9p venv   (previous)   run$i: "
  TRITON_CACHE_DIR=$COLD "$OLD" "$B"
done
