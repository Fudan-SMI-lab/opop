"""GPU-side check of the cosine overflow fix (torch required, so not in pytest on host).

Mirrors the four assertions added to tests/test_improvements.py, but imports only
worker_main so it runs in the WSL GPU venv without the host's httpx/pydantic deps.

Run: wsl.exe -d Ubuntu -- bash -lc "PYTHONPATH=/mnt/d/.../v2/src \
       ~/kernel-opt-venv/bin/python scripts/check_cosine_fix.py"
"""

import importlib.util
import sys

import torch

spec = importlib.util.find_spec("kernel_optimizer.gpu.worker_main")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_relaxed_close = mod._relaxed_close
_cos = mod._cosine_similarity

FAILS: list[str] = []


def check(name: str, cond: bool) -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


big = torch.full((4096,), 1e22)

# The bug: fp32 norm overflows at this magnitude, so the old formula produced nan.
check("fp32 norm really does overflow at 1e22 (the bug's precondition)",
      not torch.isfinite(big.flatten().norm()))

check("identical 1e22 tensors pass the gate",
      _relaxed_close(big, big.clone(), 0.01, 0.99, 0.99985))
check("cosine of identical 1e22 tensors is 1.0",
      abs(_cos(big, big.clone()) - 1.0) < 1e-6)
check("cosine stays within [-1, 1]", -1.0 <= _cos(big, big.clone()) <= 1.0)
check("opposed 1e22 tensors give cosine -1", abs(_cos(big, -big) + 1.0) < 1e-6)
check("all-zero pair is 1.0, not nan", _cos(torch.zeros(10), torch.zeros(10)) == 1.0)

# The fix must not make the gate permissive.
wrong = big.clone()
wrong[:1000] = -1e22
check("24% sign-flipped at 1e22 is still REJECTED",
      not _relaxed_close(big, wrong, 0.01, 0.99, 0.99985))
check("all-zero output vs 1e22 reference is still REJECTED",
      not _relaxed_close(big, torch.zeros_like(big), 0.01, 0.99, 0.99985))

# The regression the fix was found by: level3/48's real rejected witness profile.
# frac_within_1% = 0.999983 with median rel err 4e-7 must be accepted.
ref = torch.full((100000,), 1e22)
got = ref.clone()
got[:17] = 5e21  # 0.0017% of elements badly wrong => frac 0.99983
check("level3/48 witness profile (frac 0.99983 at 1e22) is ACCEPTED",
      _relaxed_close(ref, got, 0.01, 0.99, 0.99985))

m = mod._relaxed_metrics(ref, got)
check("failure metrics report the gate's real criteria",
      all(k in m for k in ("frac_within_tol", "cosine", "median_rel_err",
                           "p99_rel_err", "max_abs_diff", "ref_absmax")))
check("reported cosine is not nan", m["cosine"] != "nan")
print("\nmetrics sample:", m)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
