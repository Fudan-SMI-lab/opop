"""Is the harness's torch_compile_tf32 baseline slow because 5 warmup iterations is too few?

Three independent fresh measurements put `torch_compile_tf32` on L3:21 at 15.08, 15.10 and 14.96
ms, while the harness recorded 16.4 -- about 8% slower, consistently. Every speedup the run
reports against that baseline is inflated by roughly that much, so the cause matters.

The harness delegates baselines to KernelBench's `measure_ref_program_time`, whose default is
`num_warmup=5, discard_first=1`. My own measurements used 20 warmup iterations. For an eager model
5 is plenty; for `torch.compile` the first iterations pay compilation, and inductor may also
re-tune, so 5 may not be enough to reach steady state.

This sweeps warmup counts on the same model in one process and reports the resulting mean. If the
mean falls as warmup rises and plateaus near 15, the harness's 16.4 is a warmup artefact. If it is
flat, the cause is elsewhere and this hypothesis is wrong.

Run inside the WSL venv:
  python scripts/audit_baseline_warmup_sensitivity.py <reference.py>
"""
from __future__ import annotations

import importlib.util
import statistics
import sys

import torch

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")

SEED = 42
TRIALS = 100


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def time_model(model, inputs, warmup: int, trials: int, discard_first: int = 1) -> dict:
    for _ in range(warmup):
        model(*inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials + discard_first):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(*inputs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples = samples[discard_first:]
    return {"mean": statistics.mean(samples), "median": statistics.median(samples),
            "min": min(samples), "max": max(samples), "std": statistics.stdev(samples)}


def main() -> int:
    ref_path = sys.argv[1]
    ref_mod = load(ref_path, "refmod")
    device = torch.device("cuda:0")

    set_seed(SEED)
    init = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_init_inputs()]
    set_seed(SEED)
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_inputs()]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print("torch_compile_tf32, warmup sweep, 100 timed samples each "
          "(discard_first=1, as KernelBench does)\n")
    print(f"{'warmup':>7s} {'mean':>7s} {'median':>7s} {'min':>7s} {'max':>7s} {'std':>6s}")

    with torch.no_grad():
        # One compiled instance, warmed progressively: this isolates warmup from
        # compilation cost, which only the first instance pays.
        set_seed(SEED)
        model = torch.compile(ref_mod.Model(*init).to(device))
        cumulative = 0
        for target in (5, 10, 20, 40, 80):
            extra = target - cumulative
            s = time_model(model, inputs, warmup=extra, trials=TRIALS)
            cumulative = target
            print(f"{target:7d} {s['mean']:7.2f} {s['median']:7.2f} {s['min']:7.2f} "
                  f"{s['max']:7.2f} {s['std']:6.2f}")

        # And a FRESH compiled instance at warmup=5, which is what the harness does
        # (new process, new compile, 5 warmup).
        print("\nfresh torch.compile instance at warmup=5 (the harness's situation):")
        set_seed(SEED)
        fresh = torch.compile(ref_mod.Model(*init).to(device))
        s = time_model(fresh, inputs, warmup=5, trials=TRIALS)
        print(f"{5:7d} {s['mean']:7.2f} {s['median']:7.2f} {s['min']:7.2f} "
              f"{s['max']:7.2f} {s['std']:6.2f}")

        print("\nand eager_tf32 for contrast (should be warmup-insensitive):")
        set_seed(SEED)
        eager = ref_mod.Model(*init).to(device)
        for target in (5, 20):
            s = time_model(eager, inputs, warmup=target, trials=TRIALS)
            print(f"{target:7d} {s['mean']:7.2f} {s['median']:7.2f} {s['min']:7.2f} "
                  f"{s['max']:7.2f} {s['std']:6.2f}")

    print("\nThe harness recorded 16.4 ms. If the warmup=5 rows land near 16 and the")
    print("higher-warmup rows near 15, too little warmup explains the gap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
