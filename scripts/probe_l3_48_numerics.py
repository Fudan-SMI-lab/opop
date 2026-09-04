"""Is level3/48 numerically comparable at all?

Every L3:48 seed and every repair failed the default witness with max-abs-diff 1e15
to 1e22. Before blaming the candidates, establish whether the REFERENCE is even
self-consistent: the model holds A = nn.Parameter(torch.randn(...)) and computes
exp(cumsum(A)) over block_len=64, so A_cumsum ranges over roughly +/-8 and the
exponentials reach e^16 ~ 9e6 before four chained einsums over 2048 batches. If the
reference's own output changes by more than the relaxed gate's tolerance under a
mathematically equivalent reordering, then no fused kernel can ever pass and the task
is unusable for correctness comparison -- a task-selection fact, not a harness bug.

Three probes, all against the same weights and inputs:
  fp64   -- the trustworthy value; fp32-vs-fp64 shows the reference's own error
  compile-- torch.compile reorders/fuses exactly like a candidate would
  perm   -- summing the chunk recurrence in a different order

Run:  wsl.exe -d Ubuntu -- bash -lc "TRITON_CACHE_DIR=~/.triton-cache-kopt \
        ~/kernel-opt-venv/bin/python /mnt/d/.../probe_l3_48_numerics.py"
"""

import importlib.util
import sys

import torch

REF = "/mnt/d/Pyhon_projects/opop/KernelBench/KernelBench/level3/48_Mamba2ReturnY.py"

# The harness's relaxed gate, mirrored from worker_main.run_relaxed_correctness.
ELEM_TOL = 0.01
PASS_FRAC = 0.99
COSINE_MIN = 0.99985


def relaxed_close(ref: torch.Tensor, got: torch.Tensor) -> tuple[bool, dict]:
    if ref.shape != got.shape:
        return False, {"reason": "shape"}
    r = ref.flatten().double()
    g = got.flatten().double()
    denom = r.abs().clamp_min(1e-8)
    rel = (g - r).abs() / denom
    frac = (rel <= ELEM_TOL).double().mean().item()
    cos = torch.nn.functional.cosine_similarity(r, g, dim=0).item()
    ok = frac >= PASS_FRAC and cos >= COSINE_MIN
    return ok, {
        "frac_within_1pct": round(frac, 6),
        "cosine": round(cos, 8),
        "max_abs_diff": f"{(g - r).abs().max().item():.3e}",
        "median_rel": f"{rel.median().item():.3e}",
    }


def main() -> None:
    spec = importlib.util.spec_from_file_location("ref48", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    dev = "cuda"
    model = mod.Model(*mod.get_init_inputs()).to(dev)
    x = mod.get_inputs()[0].to(dev)

    print(f"A: mean={model.A.mean().item():+.4f} std={model.A.std().item():.4f} "
          f"min={model.A.min().item():+.3f} max={model.A.max().item():+.3f}")
    a_cum = torch.cumsum(model.A.reshape(model.A.shape[0], -1, 64, model.A.shape[-1]), dim=2)
    print(f"cumsum(A) over a block: min={a_cum.min().item():+.2f} "
          f"max={a_cum.max().item():+.2f}  ->  exp() spans "
          f"{torch.exp(a_cum.min()).item():.2e} .. {torch.exp(a_cum.max()).item():.2e}")

    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        y32 = model(x).float()
        print(f"\nreference fp32 output: absmax={y32.abs().max().item():.3e} "
              f"absmedian={y32.abs().median().item():.3e}")

        # 1. fp32 vs fp64: the reference's OWN numerical error.
        y64 = model.double()(x.double())
        model.float()
        ok, m = relaxed_close(y64, y32.double())
        print(f"\n[{'PASS' if ok else 'FAIL'}] reference fp32 vs its own fp64 value: {m}")

        # 2. torch.compile: a legitimate fusion/reordering, exactly what a candidate does.
        try:
            compiled = torch.compile(model)
            yc = compiled(x).float()
            ok2, m2 = relaxed_close(y32, yc)
            print(f"[{'PASS' if ok2 else 'FAIL'}] reference fp32 vs torch.compile: {m2}")
        except Exception as exc:  # compile is optional evidence
            print(f"[ N/A] torch.compile probe failed: {type(exc).__name__}: {exc}")

        # 3. tf32: the second witness precision the relaxed gate accepts.
        torch.backends.cuda.matmul.allow_tf32 = True
        ytf = model(x).float()
        ok3, m3 = relaxed_close(y32, ytf)
        print(f"[{'PASS' if ok3 else 'FAIL'}] reference fp32 vs same code in tf32: {m3}")
        torch.backends.cuda.matmul.allow_tf32 = False

    print("\nVERDICT: if the fp64 or torch.compile rows FAIL, the reference is not\n"
          "self-consistent under reordering and no candidate can pass this gate.")


if __name__ == "__main__":
    sys.exit(main())
