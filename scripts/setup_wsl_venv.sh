#!/usr/bin/env bash
# Set up the WSL GPU venv for kernel-optimizer.
# Run inside WSL:  bash /mnt/d/Pyhon_projects/opop/v2/scripts/setup_wsl_venv.sh
# Ends with a REAL triton kernel launch probe on the GPU (sm_120 gate).
#
# IMPORTANT — put the venv on ext4, never under /mnt/*. The Windows drives are
# mounted over 9p, where small-file reads cost ~100x more than on native ext4.
# `import torch` alone reads thousands of files, so a /mnt-hosted venv made every
# one-shot worker pay ~26.8s of startup; from ext4 the same work takes ~2.4s
# (measured 3x alternating, incl. a real triton compile+launch). With ~1000 GPU
# jobs per L3 run that is the difference between 8.1h and 1.4h of GPU wall clock.
# The default below ($HOME) is on ext4, which lives in the WSL ext4.vhdx on C:.

set -euo pipefail

VENV="${KERNEL_OPT_VENV:-$HOME/kernel-opt-venv}"
PYTHON="${PYTHON:-python3}"

case "$VENV" in
  /mnt/*)
    echo "REFUSING: $VENV is on a 9p Windows mount, which costs ~26s per worker" >&2
    echo "startup. Use a path on ext4 (e.g. \$HOME/kernel-opt-venv)." >&2
    exit 1
    ;;
esac

echo "== creating venv at $VENV"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Deliberately lean: only what worker_main.py imports. The old kernelfoundry venv
# carried openvino/nicegui/transformers/pandas/scipy (~1.5G) that the worker never
# touches, and every byte on C: matters here.
echo "== installing torch 2.9 (cu129) + triton 3.5 + deps"
pip install --upgrade pip -q
pip install -q "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu129
pip install -q "triton==3.5.*" ninja numpy pydantic einops


echo "== probing CUDA"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in WSL"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0),
      "| capability:", torch.cuda.get_device_capability(0))
EOF

echo "== probing kernelbench import"
PYTHONPATH=/mnt/d/Pyhon_projects/opop/KernelBench/src python - <<'EOF'
import kernelbench
from kernelbench.eval import eval_kernel_against_ref  # noqa: F401
from kernelbench.timing import measure_ref_program_time  # noqa: F401
from kernelbench.kernel_static_checker import validate_kernel_static  # noqa: F401
print("kernelbench importable OK")
EOF

echo "== REAL triton kernel launch probe"
python - <<'EOF'
import torch, triton, triton.language as tl

@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
             tl.load(y_ptr + offs, mask=mask), mask=mask)

n = 1 << 20
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
grid = (triton.cdiv(n, 1024),)
_add[grid](x, y, out, n, BLOCK=1024)
torch.cuda.synchronize()
assert torch.allclose(out, x + y), "triton kernel produced wrong result"

# resource metadata readable (duck-typed pattern used by the harness)
entry = _add.device_caches[0]
compiled = list(entry[0].values())[0]
print("triton", triton.__version__, "| launch OK | n_regs:", compiled.n_regs,
      "| shared:", compiled.metadata.shared)
EOF

echo "== ALL GREEN. venv: $VENV"
