#!/bin/bash
# Real-GPU check of the anti-reward-hacking guard in the relaxed correctness path.
# Runs two candidates against KernelBench level1/19 (ReLU):
#   honest_kernel.py   -> a genuine torch.relu; must be ACCEPTED
#   cheating_kernel.py -> returns an uninitialized buffer; must be REJECTED
# Usage (from Windows):  wsl.exe -d Ubuntu -- bash scripts/check_antihack.sh
set -u
ROOT=/mnt/d/Pyhon_projects/opop/v2
REF=/mnt/d/Pyhon_projects/opop/KernelBench/KernelBench/level1/19_ReLU.py
V=$HOME/kernel-opt-venv/bin/python
OUT=$(mktemp -d)

run_one() {
  name=$1; kernel=$2
  cat > "$OUT/job.json" <<JSON
{"job_type": "eval_correctness_relaxed",
 "ref_src_path": "$REF",
 "kernel_src_path": "$kernel",
 "num_correct_trials": 3,
 "num_perf_trials": 20,
 "backend": "triton",
 "precision": "fp32",
 "seed": 0,
 "collect_triton_metadata": false,
 "relaxed_elem_tol": 0.01,
 "relaxed_pass_frac": 0.99,
 "cosine_min": 0.99985,
 "excessive_speedup_threshold": 10.0}
JSON
  TRITON_CACHE_DIR=$HOME/.triton-cache-kopt \
  PYTHONPATH=/mnt/d/Pyhon_projects/opop/KernelBench/src \
    "$V" "$ROOT/src/kernel_optimizer/gpu/worker_main.py" \
      --job "$OUT/job.json" --out "$OUT/$name.json" >/dev/null 2>&1
  "$V" - "$OUT/$name.json" "$name" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("%-9s ok=%-5s kind=%-18s speedup=%s" % (
    sys.argv[2], r.get("ok"), r.get("failure_kind"),
    ("%.1fx" % r["speedup_vs_ref_in_worker"]) if r.get("speedup_vs_ref_in_worker") else "n/a"))
if not r.get("ok"):
    tail = str(r.get("log_tail", ""))[-500:].replace("\n", " | ")
    print("   log:", tail)
PY
}

run_one honest "$ROOT/tests/fixtures/honest_kernel.py"
run_one cheat  "$ROOT/tests/fixtures/cheating_kernel.py"
run_one timing "$ROOT/tests/fixtures/timing_cheat_kernel.py"
rm -rf "$OUT"
