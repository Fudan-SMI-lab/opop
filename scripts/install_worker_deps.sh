#!/bin/bash
# Install the deps kernelbench.utils imports at module scope. worker_main stubs only
# litellm; dotenv/tqdm/openai are imported for real and must be present or
# `import kernelbench` fails before any eval can run.
set -u
V="$HOME/kernel-opt-venv"
"$V/bin/pip" install -q python-dotenv tqdm openai requests 2>&1 | tail -5
echo "--- verify all worker deps ---"
"$V/bin/python" - <<'PY'
mods = ["torch", "triton", "numpy", "einops", "pydantic", "dotenv", "tqdm", "openai",
        "requests"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append("%s (%s)" % (m, type(e).__name__))
print("MISSING:", missing if missing else "none")
PY
echo "--- import kernelbench the way worker_main does (litellm stubbed) ---"
PYTHONPATH=/mnt/d/Pyhon_projects/opop/KernelBench/src "$V/bin/python" - <<'PY'
# worker_main._ensure_optional_deps stubs litellm: kernelbench.utils imports it at
# module scope for LLM helpers this worker never calls. Mirror that here, otherwise
# this check fails on a dependency the real worker does not need.
import importlib.util
import sys
import types

if importlib.util.find_spec("litellm") is None:
    stub = types.ModuleType("litellm")
    stub.completion = None
    sys.modules["litellm"] = stub

import kernelbench  # noqa: F401
from kernelbench.eval import eval_kernel_against_ref  # noqa: F401
from kernelbench.kernel_static_checker import validate_kernel_static  # noqa: F401
from kernelbench.timing import measure_ref_program_time  # noqa: F401

print("kernelbench importable OK (litellm stubbed, as in worker_main)")
PY
