"""P5 的可行性取决于一个数:一次 compile probe 多贵?

如果把 screen 前移进 ask() 的 guard 回路,每个被拒的采样都要付一次 probe。ask() 最坏会连拒
max_guard_rejects_per_ask=64 次,所以 64 x probe 就是一次 ask 的最坏延迟。这个数决定 P5 是
"直接前移"还是"必须先加一层便宜的预筛"。

同时测缓存命中的收益:同一个 materialized source 第二次应该几乎免费。
"""
import json, pathlib, subprocess, sys, time, re
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_compile_probe_job

REF = ("/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/"
       "43_MinGPTCausalAttention.py")
BASE = pathlib.Path("/root/autodl-tmp/ext-eval/fp32/k_tf32_128_32_2.py").read_text()
W = pathlib.Path("/root/autodl-tmp/ext-eval/probecost")
W.mkdir(exist_ok=True)
ENV = {"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:/usr/local/bin:"
               "/usr/bin:/bin", "HOME": "/root", "CUDA_HOME": "/usr/local/cuda",
       "TORCH_CUDA_ARCH_LIST": "8.9",
       "PYTHONPATH": "/root/autodl-tmp/opop-workspace/KernelBench/src"}

def build(compute, bm, bn, stages):
    s = BASE
    for k, v in {"ATTN_BLOCK_M": bm, "ATTN_BLOCK_N": bn, "ATTN_NUM_STAGES": stages,
                 "COMPUTE_DTYPE": f"'{compute}'",
                 "IO_DTYPE": "'fp16'" if compute == "fp16" else "'fp32'"}.items():
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {v}", s); assert n == 1, k
    return s

def run_probe(path, tag):
    job = make_compile_probe_job(REF, str(path), backend="triton")
    jp = W / f"{tag}.job.json"; jp.write_text(json.dumps(job))
    out = W / f"{tag}.out.json"
    t0 = time.perf_counter()
    subprocess.run(["/root/autodl-tmp/kernel-opt-venv/bin/python",
                    "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/"
                    "worker_main.py", "--job", str(jp), "--out", str(out)],
                   capture_output=True, text=True, timeout=900, env=ENV)
    dt = time.perf_counter() - t0
    d = json.load(open(out)) if out.exists() else {}
    return dt, d.get("ok"), d.get("max_shared")

configs = [("fp16", 64, 32, 1), ("fp16", 128, 64, 3), ("tf32", 128, 32, 2),
           ("ieee", 128, 64, 2), ("tf32", 64, 64, 3)]
times = []
print("cold probes (one worker process each, as the harness runs them):")
for i, (c, bm, bn, st) in enumerate(configs):
    p = W / f"c{i}.py"; p.write_text(build(c, bm, bn, st))
    dt, ok, ms = run_probe(p, f"c{i}")
    times.append(dt)
    print(f"  {c:5s} BM={bm:3d} BN={bn:2d} stg={st}  {dt:6.2f}s  ok={ok} max_shared={ms}")

import statistics
med = statistics.median(times)
print(f"\n  median cold probe: {med:.2f}s   min {min(times):.2f}  max {max(times):.2f}")
print(f"  a 64-reject ask() at this cost = {64 * med / 60:.1f} min WORST CASE")
print(f"  a full 40-trial space, all screened = {40 * med / 60:.1f} min")
print(f"\n  for scale: L3:43 spent 0.93 h on 180 shared-memory trials, i.e. "
      f"{0.93*3600/180:.1f}s per wasted trial (materialize + launch + failure)")
