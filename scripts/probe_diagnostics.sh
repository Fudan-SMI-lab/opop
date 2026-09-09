#!/bin/bash
# Feasibility probe for counter-free bottleneck diagnostics.
# Answers, on THIS box, whether each technique is available. Every check prints a verdict line.
echo "=== box: $(hostname)  date: $(date -Iseconds)"
echo "=== GPU"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem --format=csv 2>&1 | head -3

echo
echo "=== 1. can we LOCK the SM clock? (nvidia-smi -lgc)"
nvidia-smi -lgc 500,500 2>&1 | head -3
echo "--- reading back:"
nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader 2>&1 | head -2
echo "--- resetting:"
nvidia-smi -rgc 2>&1 | head -2

echo
echo "=== 2. can we LOCK the MEMORY clock? (nvidia-smi -lmc)"
nvidia-smi -lmc 5001,5001 2>&1 | head -3
echo "--- resetting:"
nvidia-smi -rmc 2>&1 | head -2

echo
echo "=== 3. persistence / permission context"
echo "uid=$(id -u)  caps:"
grep -E 'CapEff|CapPrm' /proc/self/status 2>/dev/null
echo "in container? $(test -f /.dockerenv && echo yes || echo unknown)"

echo
echo "=== 4. is ncu present, and what does it say (expect ERR_NVGPUCTRPERM)"
which ncu 2>&1 | head -1

echo
echo "=== 5. NVBit / CUPTI availability"
ls /usr/local/cuda/extras/CUPTI/lib64/ 2>&1 | head -5
python3 -c "print('no python3')" 2>/dev/null || echo "(python3 absent; use venv python)"

echo
echo "=== 6. does the venv torch expose launch/grid info + peak memory?"
/root/autodl-tmp/kernel-opt-venv/bin/python - <<'PY' 2>&1 | head -20
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("max_memory_allocated available:", hasattr(torch.cuda, "max_memory_allocated"))
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:
    print("triton import failed:", e)
# Does a compiled triton kernel expose grid / launch metadata?
try:
    import triton.language as tl
    @triton.jit
    def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)
    x = torch.randn(4096, device="cuda"); y = torch.empty_like(x)
    compiled = k[(4,)](x, y, x.numel(), BLOCK=1024)
    print("compiled type:", type(compiled).__name__)
    md = getattr(compiled, "metadata", None)
    print("metadata fields:", sorted(md._asdict().keys()) if hasattr(md, "_asdict") else type(md))
    for attr in ("n_regs", "n_spills", "shared", "num_warps", "num_ctas", "grid"):
        print("  ", attr, "=", getattr(compiled, attr, getattr(md, attr, "ABSENT")))
except Exception as e:
    print("triton probe failed:", type(e).__name__, e)
PY
echo
echo "=== DONE"
