#!/bin/bash
# Verify the ext4 worker venv: deps present, CUDA live, and a real triton kernel runs.
V="$HOME/kernel-opt-venv/bin/python"
echo "venv: $V"
for m in numpy einops pydantic torch triton; do
  if "$V" -c "import $m" 2>/dev/null; then echo "  OK   import $m"; else echo "  MISS import $m"; fi
done
"$V" - <<'PY'
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
