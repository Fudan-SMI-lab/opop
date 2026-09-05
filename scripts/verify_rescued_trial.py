"""Verify a candidate the fp64 relative gate rescued, in a fresh process.

The falsification rule in `docs/plan-next-round-four-fixes.md` says a rescued candidate that
then fails a full re-eval means the multiplier is too loose. This runs the same comparison
the worker runs, independently, so a rescue can be checked the moment it happens instead of
waiting hours for the run's own final re-eval.

It reports, per trial:
  - the ABSOLUTE gate against both reference precisions (the arm that failed);
  - the REFERENCE's own ieee-vs-tf32 agreement, i.e. the task's noise floor;
  - RMSE against an fp64 golden for the ieee reference, the tf32 reference and the
    candidate, and the candidate/reference ratio the gate actually thresholds.

Two things it deliberately mirrors rather than improves:
  - No `.eval()`. The worker does not call it, so BatchNorm runs in TRAIN mode on batch
    statistics. Calling it here would compare against a different function than the
    harness does (see `opop v2 mbconv train-mode BN`).
  - `_relaxed_close`'s exact numerics, `(ref-got).abs() / (ref.abs() + 1e-7)`, not a
    clamped variant -- a different denominator moves `frac_within_tol` in the third
    decimal, which is the range the whole question turns on.

Usage:
  python scripts/verify_rescued_trial.py <materialized_trial.py> <reference.py> [n_trials]

Run it inside the WSL venv (it needs torch + CUDA).
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


def set_prec(mode: str) -> None:
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def rmse(ref: torch.Tensor, got: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(
        ref.to(torch.float64) - got.to(torch.float64)))).item()


def relaxed(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, bool]:
    r = ref.to(torch.float32)
    g = got.to(torch.float32)
    rel = (r - g).abs() / (r.abs() + 1e-7)
    frac = (rel < ELEM_TOL).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        r.flatten().double(), g.flatten().double(), dim=0).item()
    return frac, cos, (frac > PASS_FRAC and cos >= COSINE_MIN)


def main() -> int:
    kernel_path, ref_path = sys.argv[1], sys.argv[2]
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    ref_mod = load(ref_path, "refmod")
    k_mod = load(kernel_path, "kmod")
    device = torch.device("cuda:0")

    set_seed(SEED)
    init = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_init_inputs()]
    with torch.no_grad():
        set_seed(SEED)
        ref_model = ref_mod.Model(*init)
        set_seed(SEED)
        cand_model = k_mod.ModelNew(*init)
        set_seed(SEED)
        golden_model = ref_mod.Model(*[
            x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in init
        ]).to(device=device, dtype=torch.float64)

    torch.manual_seed(SEED)
    trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(n_trials)]

    print(f"kernel : {kernel_path.split('/')[-1]}")
    print(f"gate   : frac > {PASS_FRAC} AND cosine >= {COSINE_MIN}, elem_tol {ELEM_TOL}\n")

    ratios: list[float] = []
    abs_passes = 0
    with torch.no_grad():
        for i, ts in enumerate(trial_seeds):
            set_seed(ts)
            inputs = [x.to(device) if torch.is_tensor(x) else x
                      for x in ref_mod.get_inputs()]
            set_seed(ts)
            model = ref_model.to(device=device, dtype=torch.float32)
            set_seed(ts)
            model_new = cand_model.to(device=device, dtype=torch.float32)

            set_prec("tf32")
            out_tf32 = model(*inputs)
            torch.cuda.synchronize()
            set_prec("ieee")
            out_ieee = model(*inputs)
            torch.cuda.synchronize()
            out_cand = model_new(*inputs)
            torch.cuda.synchronize()

            f_t, c_t, ok_t = relaxed(out_tf32, out_cand)
            f_i, c_i, ok_i = relaxed(out_ieee, out_cand)
            abs_ok = ok_t or ok_i
            abs_passes += int(abs_ok)

            golden = golden_model(*[x.double() if torch.is_tensor(x) else x
                                    for x in inputs])
            r_tf32 = rmse(golden, out_tf32)
            r_ieee = rmse(golden, out_ieee)
            r_cand = rmse(golden, out_cand)
            ratio = r_cand / r_tf32 if r_tf32 else float("nan")
            ratios.append(ratio)
            f_floor, _, _ = relaxed(out_ieee, out_tf32)

            print(f"trial {i}:  ABSOLUTE gate -> {'PASS' if abs_ok else 'FAIL'}")
            print(f"   vs tf32 ref : frac={f_t:.6f} cosine={c_t:.9f}")
            print(f"   vs ieee ref : frac={f_i:.6f} cosine={c_i:.9f}")
            print(f"   reference's OWN ieee-vs-tf32 floor : frac={f_floor:.6f}")
            print(f"   RMSE vs fp64: ieee-ref {r_ieee:.4e}  tf32-ref {r_tf32:.4e}  "
                  f"cand {r_cand:.4e}")
            note = "   (MORE accurate than the reference)" if ratio <= 1.0 else ""
            print(f"   candidate / tf32-reference error ratio = {ratio:.4f}{note}")
            for mult in (1.0, 2.0, 3.0):
                thr = mult * r_tf32 + ELEM_TOL / 10.0
                print(f"      fp64 relative @ {mult}: {r_cand:.4e} <= {thr:.4e} -> "
                      f"{'PASS' if r_cand <= thr else 'FAIL'}")
            print()

    good = [r for r in ratios if r == r]
    print(f"absolute gate passed {abs_passes}/{n_trials} trials")
    if good:
        print(f"ratio over {len(good)} trials: min {min(good):.4f}  max {max(good):.4f}  "
              f"mean {sum(good) / len(good):.4f}")
        print(f"trials where the candidate is MORE accurate than the reference: "
              f"{sum(1 for r in good if r <= 1.0)}/{len(good)}")
        print(f"smallest multiplier that would admit every trial: "
              f"{max(good):.4f} (configured: 2.0 / 3.0 for low precision)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
