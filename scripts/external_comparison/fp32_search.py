"""Find the best fp32/tf32 configuration OUR Triton candidate can actually reach.

theta_best's tile was chosen under fp16, where an element is 2 bytes. At fp32 the same tile
needs 131072 bytes of shared against a 101376 limit -- so "Triton cannot do fp32 here" would be
the wrong conclusion from that failure: the TILE has to shrink, which is exactly what the
tuner would have done had it been searching at fp32.

Uses the compile-only screen (P1) to skip configurations that cannot launch, so this costs
compiles rather than launches.
"""
import itertools, json, pathlib, subprocess, sys
sys.path.insert(0, "/root/autodl-tmp/opop-workspace/opop/src")
from kernel_optimizer.gpu.jobs import make_compile_probe_job, make_relaxed_correctness_job
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace import materializer

RUN = pathlib.Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-43-20260908-053708")
REF = "/root/autodl-tmp/opop-workspace/KernelBench/KernelBench/level3/43_MinGPTCausalAttention.py"
SRC = (RUN / "candidates" / "cand-fbd1988e" / "source.py").read_text(encoding="utf-8")
WORK = pathlib.Path("/root/autodl-tmp/ext-eval/fp32"); WORK.mkdir(parents=True, exist_ok=True)
ENV = {"PATH": "/root/autodl-tmp/kernel-opt-venv/bin:/usr/local/cuda/bin:/usr/local/bin:"
               "/usr/bin:/bin", "HOME": "/root", "CUDA_HOME": "/usr/local/cuda",
       "PYTHONPATH": "/root/autodl-tmp/opop-workspace/KernelBench/src"}
WORKER = "/root/autodl-tmp/opop-workspace/opop/src/kernel_optimizer/gpu/worker_main.py"

def run(job, tag):
    jp = WORK / f"job_{tag}.json"; jp.write_text(json.dumps(job, indent=1))
    op = WORK / f"out_{tag}.json"
    subprocess.run(["/root/autodl-tmp/kernel-opt-venv/bin/python", WORKER,
                    "--job", str(jp), "--out", str(op)],
                   capture_output=True, text=True, timeout=3600, env=ENV)
    return json.load(open(op))

BASE = {"ATTN_NUM_WARPS": 8, "LINEAR_NUM_WARPS": 8, "LINEAR_BLOCK_K": 32,
        "LINEAR_NUM_STAGES": 2, "LINEAR_BLOCK_M": 128, "LINEAR_BLOCK_N": 128}
# Sweep the attention tile, which is what blew the shared budget at 4 bytes/element.
GRID = list(itertools.product([64, 128], [32, 64], [1, 2]))

for cdt in ("tf32", "ieee"):
    print(f"\n  === COMPUTE_DTYPE={cdt} (IO fp32) ===")
    print(f"  {'BM':>4s} {'BN':>4s} {'stg':>4s} {'shared':>8s} {'screen':10s} {'median_ms':>10s}")
    best = None
    for bm, bn, stg in GRID:
        p = dict(BASE, ATTN_BLOCK_M=bm, ATTN_BLOCK_N=bn, ATTN_NUM_STAGES=stg,
                 COMPUTE_DTYPE=cdt, IO_DTYPE="fp32")
        tag = f"{cdt}_{bm}_{bn}_{stg}"
        try:
            mat = materializer.materialize(SRC, ParamSet(values=p))
        except Exception as exc:
            print(f"  {bm:4d} {bn:4d} {stg:4d} materialize: {exc}"); continue
        kp = WORK / f"k_{tag}.py"; kp.write_text(mat, encoding="utf-8")
        probe = run(make_compile_probe_job(REF, str(kp), backend="triton"), f"p_{tag}")
        sh = probe.get("max_shared")
        if probe.get("ok") and sh and sh > 101376:
            print(f"  {bm:4d} {bn:4d} {stg:4d} {sh:8d} {'REFUSED':10s} {'-':>10s}")
            continue
        job = make_relaxed_correctness_job(
            REF, str(kp), num_correct_trials=5, backend="triton", precision="fp32", seed=0,
            collect_kernel_metadata=False, relaxed_elem_tol=0.01, relaxed_pass_frac=0.99,
            cosine_min=0.99985, fp64_relative_gate=True, fp64_rel_multiplier=2.0,
            fp64_rel_multiplier_lowp=3.0)
        job["num_perf_trials"] = 100
        d = run(job, tag)
        lat = d.get("latency_ms") or {}
        ms = lat.get("median")
        ok = d.get("correct")
        print(f"  {bm:4d} {bn:4d} {stg:4d} {str(sh or '-'):>8s} {'ok':10s} "
              f"{(f'{ms:.3f}' if ms and ok else '-'):>10s}"
              + ("" if ok else f"   [{d.get('failure_kind')}]"))
        if ok and ms and (best is None or ms < best[0]):
            best = (ms, bm, bn, stg)
    if best:
        print(f"  BEST {cdt}: {best[0]:.3f} ms at BM={best[1]} BN={best[2]} stages={best[3]}"
              f"   -> vs external 8.675 ms: {8.674816/best[0]:.2f}x")
