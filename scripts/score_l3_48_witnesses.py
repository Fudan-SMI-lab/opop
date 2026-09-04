"""Score the rejected L3:48 witnesses on the metrics the gate actually uses.

Five rejections reported only max-abs-diff, and the value 7318349394477056 recurs
identically across two different candidates and three different repairs -- which smells
like the reference's own numerical spread rather than a per-candidate bug. This measures
each rejected witness against both reference precisions with frac-within-tol and cosine,
alongside the reference's own ieee-vs-tf32 spread as the noise floor.

If a candidate's frac_within_tol is ~0.978 (matching the reference's own ieee-vs-tf32
row from probe_l3_48_numerics.py) then it is as close to the reference as the reference
is to itself, and the 0.99 pass_frac is simply unreachable for this task.

Run: wsl.exe -d Ubuntu -- bash -lc "... python scripts/score_l3_48_witnesses.py"
"""

import glob
import importlib.util
import sys

import torch

REF = "/mnt/d/Pyhon_projects/opop/KernelBench/KernelBench/level3/48_Mamba2ReturnY.py"
RUN = "/mnt/d/Pyhon_projects/opop/v2/runs/run-l3-48-20260905-003307"


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics(ref: torch.Tensor, got: torch.Tensor) -> str:
    if ref.shape != got.shape:
        return f"SHAPE {tuple(ref.shape)} vs {tuple(got.shape)}"
    r, g = ref.float(), got.float()
    rel = (r - g).abs() / (r.abs() + 1e-7)
    a, b = r.flatten(), g.flatten()
    den = (a.norm() * b.norm()).item()
    cos = torch.dot(a, b).item() / den if den else float("nan")
    return (f"frac_within_1%={rel.lt(0.01).float().mean().item():.6f} "
            f"cos={cos:.8f} med_rel={rel.median().item():.2e} "
            f"maxabs={(r - g).abs().max().item():.3e}")


def main() -> None:
    ref_mod = load(REF, "ref48")
    torch.manual_seed(0)
    dev = "cuda"
    ref_model = ref_mod.Model(*ref_mod.get_init_inputs()).to(dev)
    x = ref_mod.get_inputs()[0].to(dev)

    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        y_ieee = ref_model(x).float()
        torch.backends.cuda.matmul.allow_tf32 = True
        y_tf32 = ref_model(x).float()
        torch.backends.cuda.matmul.allow_tf32 = False

    print("GATE: accept if frac_within_1% > 0.99 AND cos >= 0.99985 vs EITHER ref\n")
    print(f"NOISE FLOOR  ref_ieee vs ref_tf32: {metrics(y_ieee, y_tf32)}\n")

    for path in sorted(glob.glob(f"{RUN}/candidates/*/witness_default.py")):
        cid = path.split("/")[-2]
        try:
            mod = load(path, f"cand_{cid.replace('-', '_')}")
            torch.manual_seed(0)
            model = mod.ModelNew(*ref_mod.get_init_inputs()).to(dev)
            with torch.no_grad():
                out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            print(f"{cid}")
            print(f"   vs ieee: {metrics(y_ieee, out.float())}")
            print(f"   vs tf32: {metrics(y_tf32, out.float())}")
        except Exception as exc:  # a candidate may not even run
            print(f"{cid}\n   ERROR {type(exc).__name__}: {str(exc)[:200]}")

    print("\nIf a candidate's frac_within_1% matches the NOISE FLOOR row, it is as close\n"
          "to the reference as the reference is to itself -> the task cannot pass a\n"
          "0.99 frac gate, and this is task selection, not a candidate bug.")


if __name__ == "__main__":
    sys.exit(main())
