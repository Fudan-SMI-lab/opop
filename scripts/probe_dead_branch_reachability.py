"""Is the "dead branch under the test distribution" substitution reachable in our REAL tasks?

Context: on L1:19 (ReLU) the novelty agent produced a pure memcpy, arguing that `get_inputs()`
returns `torch.rand` (non-negative) so ReLU is identity on every input the harness constructs.
That argument is CORRECT for that task, and `get_inputs()` is benchmark-provided (KernelBench
pinned at 423217d, file unmodified). So the question is not "is the agent cheating" but:

  for the three L3 tasks we actually run, is any part of the reference computation dead on the
  values that actually reach it?

An input-boundary check is not enough. All four tasks feed `torch.rand` (>= 0), but an L3
reference runs that input through learned layers first, and a conv/linear with randomly
initialised (signed) weights produces signed activations. So the exposure depends on the
values reaching each nonlinearity, not on the input.

This hooks every module of each reference, runs the harness's own `get_inputs()` /
`get_init_inputs()`, and reports for each activation site whether it ever sees a value that
its nonlinearity would actually change.

Run inside WSL:  python scripts/probe_dead_branch_reachability.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

KB = pathlib.Path("/mnt/d/Pyhon_projects/opop/KernelBench/KernelBench")
TASKS = [
    ("level1/19_ReLU.py", "L1:19 ReLU"),
    ("level3/21_EfficientNetMBConv.py", "L3:21 MBConv"),
    ("level3/43_MinGPTCausalAttention.py", "L3:43 MinGPT attention"),
    ("level3/48_Mamba2ReturnY.py", "L3:48 Mamba2"),
]

# Nonlinearities whose behaviour is input-sign/range dependent, i.e. the ones a candidate
# could delete if the values reaching them never exercise the clamp.
CLAMPING = ("ReLU", "ReLU6", "Hardtanh", "Hardswish", "Hardsigmoid", "Clamp",
            "Threshold", "LeakyReLU", "ELU")


def load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"ref_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    torch.manual_seed(0)
    for rel, label in TASKS:
        path = KB / rel
        if not path.exists():
            print(f"{label}: MISSING {path}")
            continue
        mod = load(path)
        model = mod.Model(*mod.get_init_inputs())
        inputs = [t.cuda() if torch.is_tensor(t) else t for t in mod.get_inputs()]
        model = model.cuda().eval() if False else model.cuda()  # harness never calls .eval()

        seen: list[tuple[str, str, float, float, bool]] = []

        def mk_hook(name, kind):
            def hook(_m, inp):
                # MUST be a FORWARD-PRE hook and MUST clone: these modules are built with
                # `inplace=True` (KernelBench 21_EfficientNetMBConv lines 25, 31), so a
                # post-hook reads the tensor AFTER it was overwritten and reports the
                # nonlinearity's own OUTPUT range as its input -- which made every site look
                # "never sees a negative" with a suspiciously exact [0.0, 6.0] range.
                t = inp[0] if isinstance(inp, tuple) else inp
                if not torch.is_tensor(t):
                    return
                t = t.detach().clone()
                lo, hi = t.min().item(), t.max().item()
                n = t.numel()
                # Fraction the clamp actually modifies, for this module's own bounds.
                if kind in ("ReLU6", "Hardtanh"):
                    changed = ((t < 0) | (t > 6.0)).sum().item()
                else:
                    changed = (t < 0).sum().item()
                seen.append((name, kind, lo, hi, changed, n))
            return hook

        handles = []
        for name, m in model.named_modules():
            kind = type(m).__name__
            if kind in CLAMPING:
                handles.append(m.register_forward_pre_hook(mk_hook(name or "<root>", kind)))

        with torch.no_grad():
            out = model(*inputs)
        for h in handles:
            h.remove()

        print(f"\n=== {label}")
        print(f"    input range: "
              f"[{inputs[0].min().item():.4f}, {inputs[0].max().item():.4f}]"
              f"   output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        # Does the top-level forward apply a clamp directly to the input?
        if not seen:
            print("    no clamping module found (nonlinearity may be functional, "
                  "e.g. torch.relu / F.gelu -- checked below)")
        for name, kind, lo, hi, changed, n in seen:
            frac = changed / n * 100 if n else 0.0
            verdict = (f"LIVE  clamp modifies {frac:5.2f}% of elements "
                       f"-> deleting it CHANGES the result"
                       if changed else
                       "DEAD  clamp modifies 0 elements -> identity equivalent HERE")
            print(f"    {kind:10s} {name:30s} in=[{lo:9.4f},{hi:9.4f}]  {verdict}")

        # Functional nonlinearities: probe by comparing the reference against a version
        # of its own input that is forced negative. If the output is unchanged in sign
        # structure the branch was dead.
        neg = [(-t.abs() if torch.is_tensor(t) else t) for t in inputs]
        with torch.no_grad():
            out_neg = model(*neg)
        same = torch.allclose(out, out_neg, rtol=0, atol=0)
        print(f"    sign-flipped input changes the output: {not same}"
              f"   (False => the whole task is insensitive to input sign)")
        # And the decisive one for the memcpy case: is the reference itself an identity?
        with torch.no_grad():
            ident = torch.allclose(out, inputs[0], rtol=1e-6, atol=1e-6) \
                if out.shape == inputs[0].shape else False
        print(f"    reference == identity on the harness's own inputs: {ident}"
              f"   {'<-- a memcpy passes here' if ident else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
