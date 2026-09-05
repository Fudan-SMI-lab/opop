"""Independently time a candidate against the reference baselines, in a fresh process.

`tuned_ms` comes from a 20-sample quick test taken during tuning, and
`opop-v2-reeval-gap-is-the-real-number` measures it as systematically optimistic by 1.5-6.7%.
The run's own `final_reeval` settles it, but that is hours away; this gives the same answer
now, at full sample count, so a headline number can be checked when it appears rather than
believed.

Times four things in one process under identical conditions:
  - the candidate
  - eager reference (ieee) and eager reference (tf32)
  - torch.compile reference (tf32), which is the strongest baseline on L3:21

Mirrors the harness's timing discipline: CUDA events, warmup, and the candidate timed under
`highest` matmul precision (the harness sets ieee before its timed region, so a candidate
that wants low precision must ask for it inside its own kernel).

Usage (inside the WSL venv):
  python scripts/verify_candidate_timing.py <trial.py> <reference.py> [n_samples]
"""
from __future__ import annotations

import importlib.util
import statistics
import sys

import torch

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")

SEED = 42
WARMUP = 20


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


def time_it(fn, inputs, n: int) -> dict:
    for _ in range(WARMUP):
        fn(*inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {"mean": statistics.mean(samples), "std": statistics.stdev(samples),
            "min": samples[0], "max": samples[-1], "median": statistics.median(samples),
            "n": n}


def show(label: str, s: dict) -> None:
    print(f"  {label:26s} mean {s['mean']:6.2f}  median {s['median']:6.2f}  "
          f"min {s['min']:6.2f}  max {s['max']:6.2f}  std {s['std']:5.2f}")


def main() -> int:
    trial, ref_path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    ref_mod = load(ref_path, "refmod")
    k_mod = load(trial, "kmod")
    device = torch.device("cuda:0")

    set_seed(SEED)
    init = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_init_inputs()]
    set_seed(SEED)
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_inputs()]

    with torch.no_grad():
        set_seed(SEED)
        ref_model = ref_mod.Model(*init).to(device)
        set_seed(SEED)
        cand = k_mod.ModelNew(*init).to(device)
        cand.load_state_dict(ref_model.state_dict(), strict=False)

        results = {}
        print(f"timing {n} CUDA-event samples each, {WARMUP} warmup\n")

        # The candidate: harness sets ieee before the timed region.
        set_prec(False)
        results["candidate"] = time_it(cand, inputs, n)
        show("candidate", results["candidate"])

        set_prec(False)
        results["eager"] = time_it(ref_model, inputs, n)
        show("eager (ieee)", results["eager"])

        set_prec(True)
        results["eager_tf32"] = time_it(ref_model, inputs, n)
        show("eager_tf32", results["eager_tf32"])

        for tf32, label in ((True, "torch_compile_tf32"), (False, "torch_compile")):
            set_prec(tf32)
            try:
                compiled = torch.compile(ref_mod.Model(*init).to(device))
                compiled.load_state_dict(ref_model.state_dict(), strict=False)
                results[label] = time_it(compiled, inputs, n)
                show(label, results[label])
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:26s} unavailable: {type(exc).__name__}: {exc}")

    c = results["candidate"]["mean"]
    print("\nspeedup vs each baseline (mean/mean):")
    for k, v in results.items():
        if k == "candidate":
            continue
        print(f"  vs {k:24s} {v['mean'] / c:5.3f}x")

    strongest = min((v["mean"] for k, v in results.items() if k != "candidate"),
                    default=None)
    if strongest:
        name = next(k for k, v in results.items()
                    if k != "candidate" and v["mean"] == strongest)
        verdict = "BEATS" if c < strongest else "LOSES TO"
        print(f"\nstrongest baseline here: {name} at {strongest:.2f} ms")
        print(f"candidate {c:.2f} ms {verdict} it ({strongest / c:.3f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
