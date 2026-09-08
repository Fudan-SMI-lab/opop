"""Evaluate the external CUDA candidate with OUR pipeline, same convention as theta_best.

One number, one way: the same relaxed-correctness job builder, the same config values from
experiments_l3_glm_linux.yaml, 5 correctness trials + 100 timed samples, exclusive GPU.
"""
import json, pathlib, subprocess, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_relaxed_correctness_job

REF = "/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/43_MinGPTCausalAttention.py"
KERNEL = "/root/autodl-tmp/ext-eval/L3_43/wrapper_l3_43.py"

job = make_relaxed_correctness_job(
    REF, KERNEL, num_correct_trials=5, backend="cuda", precision="fp32", seed=0,
    collect_kernel_metadata=True, relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
    cosine_min=0.99985, fp64_relative_gate=True, fp64_rel_multiplier=2.0,
    fp64_rel_multiplier_lowp=3.0)
job["num_perf_trials"] = 100
job["measure_launch_overhead"] = True
jp = "/root/autodl-tmp/ext-eval/L3_43/job.json"
pathlib.Path(jp).write_text(json.dumps(job, indent=1))
print(f"  backend={job['backend']}  correct=5  perf=100  overhead=True")

r = subprocess.run(
    ["/root/autodl-tmp/kernel-opt-venv/bin/python",
     "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/worker_main.py",
     "--job", jp, "--out", "/root/autodl-tmp/ext-eval/L3_43/out.json"],
    capture_output=True, text=True, timeout=3600,
    env={"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/root",
         "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
         "PYTHONPATH": "/root/autodl-tmp/opop-workspace/KernelBench/src"})
print(f"  worker rc={r.returncode}")
d = json.load(open("/root/autodl-tmp/ext-eval/L3_43/out.json"))
for k in ("ok", "compiled", "correct", "failure_kind", "trials_passed", "trials_total",
          "fp64_rescued_trials", "excessive_speedup"):
    print(f"  {k:20s} {d.get(k)}")
lat = d.get("latency_ms") or {}
if lat:
    print(f"\n  EXTERNAL CUDA: median={lat.get('median')} mean={lat.get('mean')} "
          f"std={lat.get('std')} n={lat.get('n')}")
ref = d.get("ref_latency_ms") or {}
if ref:
    print(f"  reference     : median={ref.get('median')} n={ref.get('n')}")
cub = d.get("cubin") or {}
if cub.get("kernels"):
    print(f"  cubin: launched_filter={cub.get('launched_filter')}")
    for k in cub["kernels"][:4]:
        print(f"    {str(k.get('name'))[:44]:44s} regs={k.get('n_regs')} "
              f"spills={k.get('n_spills')} shared={k.get('shared')} "
              f"tc={(k.get('sass') or {}).get('tensor_core')}")
if not d.get("correct"):
    print(f"\n  log_tail:\n{str(d.get('log_tail'))[-1800:]}")
