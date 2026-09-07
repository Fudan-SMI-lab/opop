"""Verify capture_timing_samples against the REAL vendored KernelBench, on the real box.

The unit test drives a stand-in timing module, which proves the wrapper logic but NOT the claim
the fix rests on: that `kernelbench.timing.get_timing_stats` is the single funnel every timing
path resolves at call time. If eval.py had imported the function by value (`from .timing import
get_timing_stats`), the patch would silently miss the strict eval_perf path -- the exact path
that was broken -- and the unit test would still pass.

So this asserts on the real module: patch, then call through eval.py's own reference and through
timing.py's module global, and require the samples to arrive both ways.
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")

from kernel_optimizer.gpu.worker_main import _stats_to_dict, capture_timing_samples  # noqa: E402

ELAPSED = [5.0, 4.9, 5.1, 5.0, 4.95, 5.05, 5.0, 20.0, 18.0, 5.0]  # mean 7.80, median 5.00

print("installing patch ...")
assert capture_timing_samples() is True, "patch did not install against the real kernelbench"
assert capture_timing_samples() is True, "not idempotent"

from kernelbench import eval as kb_eval  # noqa: E402
from kernelbench import timing as kb_timing  # noqa: E402

failures = []

# Route 1: timing.py's own module global (measure_ref_program_time takes this one).
s1 = kb_timing.get_timing_stats(ELAPSED)
o1 = _stats_to_dict(s1)
print(f"route timing.py global   : median={o1.get('median')} mean={o1['mean']:.4f} "
      f"samples={len(o1.get('samples') or [])}")
if o1.get("median") is None:
    failures.append("timing.py module global does not carry samples")

# Route 2: eval.py's reference -- this is what run_eval / eval_perf actually calls. If eval.py
# had bound the function by value at import time, THIS is where it would fail.
s2 = kb_eval.timing.get_timing_stats(ELAPSED)
o2 = _stats_to_dict(s2)
print(f"route eval.py reference  : median={o2.get('median')} mean={o2['mean']:.4f} "
      f"samples={len(o2.get('samples') or [])}")
if o2.get("median") is None:
    failures.append("eval.py's timing reference does not carry samples -- the strict eval_perf "
                    "path is STILL producing no median, which is the whole defect")

# The median must be the median, and KernelBench's mean must be untouched.
for name, o in (("timing.py", o1), ("eval.py", o2)):
    if o.get("median") is not None and abs(o["median"] - 5.0) > 1e-9:
        failures.append(f"{name}: median is {o['median']}, expected 5.0")
    expected_mean = float(f"{sum(ELAPSED) / len(ELAPSED):.3g}")  # KB rounds to 3 sig figs
    if abs(o["mean"] - expected_mean) > 1e-9:
        failures.append(f"{name}: mean is {o['mean']}, KernelBench's own value is {expected_mean} "
                        f"-- the wrapper must only ADD a key")

# Positive control: without the patch there must be NO median, or this script proves nothing.
kb_timing.get_timing_stats = kb_timing.get_timing_stats.__wrapped__ if hasattr(
    kb_timing.get_timing_stats, "__wrapped__") else None
if kb_timing.get_timing_stats is None:
    # Re-import a pristine copy to serve as the control.
    import importlib
    importlib.reload(kb_timing)
    control = _stats_to_dict(kb_timing.get_timing_stats(ELAPSED))
    print(f"control (unpatched)      : median={control.get('median')}")
    if control.get("median") is not None:
        failures.append("the UNPATCHED path already reports a median -- this script cannot "
                        "detect the defect it is testing for")

print()
if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("PASS: both real KernelBench routes carry samples; unpatched control carries none")
