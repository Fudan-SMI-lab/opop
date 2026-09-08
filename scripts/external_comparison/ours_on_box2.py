"""Measure OUR best L3:21 and L3:48 candidates on THIS box, through the same job builder that
measured the external CUDA kernels.

Why this run exists: our recorded 6.92 ms (L3:21) and 1.41 ms (L3:48) were produced on other
machines, and one of them (L3:21) was explicitly never independently re-measured. Comparing those
figures against external numbers taken here would be comparing two cards. Same builder, same gate
values, 5 correctness trials + 100 timed samples, and the reference baseline comes out of the same
job so the speedup is a ratio measured in one process.
"""
import json, pathlib, subprocess, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_relaxed_correctness_job

WORK = pathlib.Path("/root/autodl-tmp/ext-eval/ours")
KB = "/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3"
CASES = [
    ("L3_21_ours", f"{KB}/21_EfficientNetMBConv.py", WORK / "our_l3_21_best.py"),
    ("L3_48_ours", f"{KB}/48_Mamba2ReturnY.py", WORK / "our_l3_48_best.py"),
]

for tag, ref, kernel in CASES:
    print(f"\n{'='*78}\n### {tag}\n{'='*78}")
    job = make_relaxed_correctness_job(
        ref, str(kernel), num_correct_trials=5, backend="triton", precision="fp32", seed=0,
        collect_kernel_metadata=True, relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
        cosine_min=0.99985, fp64_relative_gate=True, fp64_rel_multiplier=2.0,
        fp64_rel_multiplier_lowp=3.0)
    job["num_perf_trials"] = 100
    job["measure_launch_overhead"] = True
    jp = WORK / f"{tag}.job.json"
    jp.write_text(json.dumps(job, indent=1))
    out = WORK / f"{tag}.out.json"
    r = subprocess.run(
        ["/root/autodl-tmp/kernel-opt-venv/bin/python",
         "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/worker_main.py",
         "--job", str(jp), "--out", str(out)],
        capture_output=True, text=True, timeout=5400,
        env={"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:/usr/local/bin:"
                     "/usr/bin:/bin",
             "HOME": "/root", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
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
    ref_l = d.get("ref_latency_ms") or {}
    if lat:
        print(f"  OURS (triton) : median={lat.get('median')} mean={lat.get('mean')} "
              f"std={lat.get('std')} n={lat.get('n')}")
    if ref_l:
        print(f"  reference eager: median={ref_l.get('median')} n={ref_l.get('n')}")
        if lat.get("median") and ref_l.get("median"):
            print(f"  --> speedup vs eager (this box): "
                  f"{ref_l['median'] / lat['median']:.2f}x")
    cub = d.get("cubin") or {}
    if cub.get("kernels"):
        print(f"  kernels={len(cub['kernels'])} launched_filter={cub.get('launched_filter')}")
        for k in cub["kernels"][:8]:
            print(f"    {str(k.get('name'))[:44]:44s} regs={k.get('n_regs')} "
                  f"spills={k.get('n_spills')} shared={k.get('shared')} "
                  f"tc={(k.get('sass') or {}).get('tensor_core')}")
    if not d.get("correct"):
        print(f"  log_tail:\n{str(d.get('log_tail'))[-2500:]}")
