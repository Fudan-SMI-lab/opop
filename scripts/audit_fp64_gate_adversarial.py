"""Adversarial check: can the fp64 relative gate be fooled?

The gate admits a candidate whose RMSE against an fp64 golden is within `multiplier` of the
REFERENCE's own RMSE. Every live rescue so far has ratio < 1.0, i.e. the candidate is more
accurate than the reference -- which is the case it exists to admit. The question this asks
is the opposite one: what does it do to a candidate that is genuinely WRONG?

RMSE is a mean over all elements, so the worry is specific and worth testing rather than
reasoning about: a candidate correct almost everywhere but badly wrong on a few elements has
its error averaged down. If that passes, the gate is weaker than the absolute one it
supplements and the flag should come off.

Four synthetic candidates against a real reference output, all judged by the same two arms
the worker uses:

  1. tf32-class noise      -- should PASS (this is the live rescue case)
  2. one element corrupted -- the averaging worry, stated concretely
  3. 1% of elements wrong  -- a sparse-but-real bug
  4. a systematic 2% scale -- the classic silently-wrong kernel

Run inside the WSL venv.
"""
from __future__ import annotations

import importlib.util
import sys

import torch

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")

ELEM_TOL = 0.01
PASS_FRAC = 0.99
COSINE_MIN = 0.99985
SEED = 42


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def set_prec(tf32: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(
        a.to(torch.float64) - b.to(torch.float64)))).item()


def absolute_gate(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, bool]:
    r = ref.to(torch.float32)
    g = got.to(torch.float32)
    rel = (r - g).abs() / (r.abs() + 1e-7)
    frac = (rel < ELEM_TOL).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        r.flatten().double(), g.flatten().double(), dim=0).item()
    return frac, cos, (frac > PASS_FRAC and cos >= COSINE_MIN)


def main() -> int:
    ref_path = sys.argv[1]
    ref_mod = load(ref_path, "refmod")
    device = torch.device("cuda:0")

    set_seed(SEED)
    init = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_init_inputs()]
    with torch.no_grad():
        set_seed(SEED)
        model = ref_mod.Model(*init).to(device)
        set_seed(SEED)
        golden_model = ref_mod.Model(*[
            x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in init
        ]).to(device=device, dtype=torch.float64)

        set_seed(SEED)
        inputs = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_inputs()]
        set_prec(True)
        out_tf32 = model(*inputs)
        torch.cuda.synchronize()
        set_prec(False)
        out_ieee = model(*inputs)
        torch.cuda.synchronize()
        golden = golden_model(*[x.double() if torch.is_tensor(x) else x for x in inputs])

    r_ref = rmse(golden, out_tf32)
    print(f"reference RMSE vs fp64 golden : {r_ref:.6e}")
    print(f"reference ieee-vs-tf32 floor  : frac={absolute_gate(out_ieee, out_tf32)[0]:.6f}")
    print(f"output shape {tuple(out_tf32.shape)}  absmax {out_tf32.abs().max().item():.6g}\n")

    # Candidate 1: tf32-class noise -- the live rescue case.
    set_seed(7)
    cands = {
        "tf32-class noise (the live case)":
            out_ieee + torch.randn_like(out_ieee) * (out_tf32 - out_ieee).abs().mean(),
    }
    # Candidate 2: exactly one element badly wrong.
    c2 = out_ieee.clone()
    flat = c2.reshape(-1)
    flat[0] = flat[0] + 1000.0
    cands["ONE element off by +1000"] = c2
    # Candidate 3: 1% of elements wrong by 50%.
    c3 = out_ieee.clone()
    f3 = c3.reshape(-1)
    n_bad = max(1, f3.numel() // 100)
    idx = torch.randperm(f3.numel(), device=f3.device)[:n_bad]
    f3[idx] = f3[idx] * 1.5
    cands[f"1% of elements ({n_bad}) off by 50%"] = c3
    # Candidate 4: systematic 2% scale error everywhere.
    cands["systematic 2% scale error"] = out_ieee * 1.02

    print(f"{'candidate':38s} {'abs gate':>9s} {'frac':>9s} {'RMSE':>11s} "
          f"{'ratio':>8s} {'@2.0':>6s} {'@3.0':>6s}")
    for name, cand in cands.items():
        frac_t, _, ok_t = absolute_gate(out_tf32, cand)
        frac_i, _, ok_i = absolute_gate(out_ieee, cand)
        abs_ok = ok_t or ok_i
        r_c = rmse(golden, cand)
        ratio = r_c / r_ref if r_ref else float("nan")
        verdicts = []
        for mult in (2.0, 3.0):
            verdicts.append("PASS" if r_c <= mult * r_ref + ELEM_TOL / 10.0 else "FAIL")
        print(f"{name:38s} {'PASS' if abs_ok else 'FAIL':>9s} "
              f"{max(frac_t, frac_i):9.6f} {r_c:11.4e} {ratio:8.3f} "
              f"{verdicts[0]:>6s} {verdicts[1]:>6s}")

    print("\nThe gate is sound iff row 1 passes and rows 2-4 fail. A row 2-4 PASS means")
    print("RMSE averaging is hiding a real defect and fp64_relative_gate should go off.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
