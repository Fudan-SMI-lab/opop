"""L3:48 的 fp16 候选需要 5/5 fp64 救回;bf16 在 L3:43 上与 fp16 打平。
那么在 L3:48 上把 COMPUTE_DTYPE/BC_CACHE_DTYPE 换成 bf16,能否既保住速度又不再需要救回?

这是一个具体的改进方案能否落地的前提检验:如果 bf16 在这里同样近乎无代价,那么"报告 per-precision
最优 + 让搜索不要在 fp16 上早停"就有直接收益;如果 bf16 明显更慢,那结论就是这个任务确实要 fp16,
救回是必要代价 —— 两种情况给出的建议完全不同,所以必须实测。

走我们自己的 job builder,同一 relaxed+fp64 门,5 次正确性 + 100 次计时。
"""
import json, pathlib, re, subprocess, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_relaxed_correctness_job

W = pathlib.Path("/root/autodl-tmp/ext-eval/ours")
REF = ("/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/"
       "48_Mamba2ReturnY.py")
BASE = (W / "our_l3_48_best.py").read_text()

VARIANTS = {
    "fp16 (shipped)": {},
    "bf16 compute+cache": {"COMPUTE_DTYPE": "bf16", "BC_CACHE_DTYPE": "bf16"},
    "bf16 compute only": {"COMPUTE_DTYPE": "bf16"},
    "fp32 cache only": {"BC_CACHE_DTYPE": "fp32"},
}

for label, over in VARIANTS.items():
    src = BASE
    for k, v in over.items():
        src, n = re.subn(rf"'{k}': '[^']*'", f"'{k}': '{v}'", src)
        assert n == 1, f"{k} replaced {n}x"
    tag = label.split()[0] + "_" + str(abs(hash(label)) % 10000)
    p = W / f"v48_{tag}.py"
    p.write_text(src)
    job = make_relaxed_correctness_job(
        REF, str(p), num_correct_trials=5, backend="triton", precision="fp32", seed=0,
        collect_kernel_metadata=True, relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
        cosine_min=0.99985, fp64_relative_gate=True, fp64_rel_multiplier=2.0,
        fp64_rel_multiplier_lowp=3.0)
    job["num_perf_trials"] = 100
    jp = W / f"v48_{tag}.job.json"; jp.write_text(json.dumps(job, indent=1))
    out = W / f"v48_{tag}.out.json"
    r = subprocess.run(
        ["/root/autodl-tmp/kernel-opt-venv/bin/python",
         "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/worker_main.py",
         "--job", str(jp), "--out", str(out)],
        capture_output=True, text=True, timeout=3600,
        env={"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:"
                     "/usr/local/bin:/usr/bin:/bin",
             "HOME": "/root", "CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9",
             "PYTHONPATH": "/root/autodl-tmp/opop-workspace/KernelBench/src"})
    if not out.exists():
        print(f"  {label:22s} NO OUTPUT rc={r.returncode} {r.stderr[-300:]}")
        continue
    d = json.load(open(out))
    lat = d.get("latency_ms") or {}
    print(f"  {label:22s} median={str(lat.get('median'))[:6]:>7s} ms  "
          f"correct={d.get('correct')} {d.get('trials_passed')}/{d.get('trials_total')}  "
          f"rescued={d.get('fp64_rescued_trials')}  kind={d.get('failure_kind')}")
