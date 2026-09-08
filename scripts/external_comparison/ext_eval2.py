"""Evaluate the remaining two external CUDA candidates with OUR pipeline.

Same convention as theta_best and as the L3:43 external evaluation: the same relaxed-correctness
job builder, the same gate values from experiments_l3_glm_linux.yaml, 5 correctness trials + 100
timed samples, exclusive GPU. Baselines come out of the same job (ref_latency_ms), so the speedup
is a ratio of two numbers measured in one process on one card -- not a comparison against the
manifest's own search-time figure, which the README itself warns is not a GPU0 remeasurement.
"""
import json, pathlib, subprocess, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_relaxed_correctness_job

WORK = pathlib.Path("/root/autodl-tmp/ext-eval")
KB = "/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3"
CASES = [
    ("L3_21", f"{KB}/21_EfficientNetMBConv.py", WORK / "L3_21/wrapper_l3_21.py",
     WORK / "L3_21/best_kernel.cu"),
    ("L3_48", f"{KB}/48_Mamba2ReturnY.py", WORK / "L3_48/wrapper_l3_48.py",
     WORK / "L3_48/best_kernel.cu"),
]

for tag, ref, wrapper, cu in CASES:
    print(f"\n{'='*78}\n### {tag}\n{'='*78}")
    job = make_relaxed_correctness_job(
        ref, str(wrapper), num_correct_trials=5, backend="cuda", precision="fp32", seed=0,
        collect_kernel_metadata=True, relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
        cosine_min=0.99985, fp64_relative_gate=True, fp64_rel_multiplier=2.0,
        fp64_rel_multiplier_lowp=3.0)
    job["num_perf_trials"] = 100
    job["measure_launch_overhead"] = True
    jp = WORK / tag / "job.json"
    jp.write_text(json.dumps(job, indent=1))
    out = WORK / tag / "out.json"
    r = subprocess.run(
        ["/root/autodl-tmp/kernel-opt-venv/bin/python",
         "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/worker_main.py",
         "--job", str(jp), "--out", str(out)],
        capture_output=True, text=True, timeout=5400,
        env={"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:/usr/local/bin:"
                     "/usr/bin:/bin",
             "HOME": "/root", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
             "EXT_KERNEL_CU": str(cu),
             "PYTHONPATH": "/root/autodl-tmp/opop-workspace/KernelBench/src"})
    print(f"  worker rc={r.returncode}")
    if not out.exists():
        print(f"  NO OUTPUT. stderr tail:\n{r.stderr[-2500:]}")
        continue
    d = json.load(open(out))
    for k in ("ok", "compiled", "correct", "failure_kind", "trials_passed", "trials_total",
              "fp64_rescued_trials", "excessive_speedup"):
        print(f"  {k:20s} {d.get(k)}")
    lat = d.get("latency_ms") or {}
    if lat:
        print(f"  EXTERNAL CUDA : median={lat.get('median')} mean={lat.get('mean')} "
              f"std={lat.get('std')} n={lat.get('n')}")
    ref_l = d.get("ref_latency_ms") or {}
    if ref_l:
        print(f"  reference eager: median={ref_l.get('median')} n={ref_l.get('n')}")
        if lat.get("median") and ref_l.get("median"):
            print(f"  --> speedup vs eager (this box, one process): "
                  f"{ref_l['median'] / lat['median']:.2f}x")
    cub = d.get("cubin") or {}
    if cub.get("kernels"):
        print(f"  cubin: launched_filter={cub.get('launched_filter')} "
              f"n_kernels={len(cub['kernels'])}")
        for k in cub["kernels"][:6]:
            print(f"    {str(k.get('name'))[:46]:46s} regs={k.get('n_regs')} "
                  f"spills={k.get('n_spills')} shared={k.get('shared')} "
                  f"tc={(k.get('sass') or {}).get('tensor_core')}")
    if not d.get("correct"):
        print(f"  log_tail:\n{str(d.get('log_tail'))[-2500:]}")
