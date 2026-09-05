"""Is cand-80bf3097's fp16 dot path actually exercised, or is the knob inert?

The suspicious observation: this candidate's RMSE against an fp64 golden matched the tf32
reference's to five significant digits (7.0296e-04 vs 7.0296e-04) with cosine 1.000000000,
on a trial whose PARAMS say COMPUTE_DTYPE='fp16'. fp16 and tf32 have different error
characteristics (fp16: 10-bit mantissa, 5-bit exponent; tf32: 10-bit mantissa, 8-bit
exponent), so identical error is not what a genuinely fp16 dot should produce against a
tf32 reference.

Three explanations, and they need separating:
  1. The knob is inert -- the fp16 branch is not the one that runs (a dead-branch bug of
     the kind `opop-v2-dead-mode-branch-strands-optimization` records).
  2. The dot is fp16 but contributes so little to total error that BN dominates it.
  3. Coincidence at 5 digits, which would be remarkable.

This materializes the SAME candidate at every COMPUTE_DTYPE value and compares outputs. If
the knob is live, the four outputs must differ from each other. If fp16 and tf32 produce
bit-identical results, the knob is not doing what its name says.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/mnt/d/Pyhon_projects/opop/KernelBench/src")

TRIAL = sys.argv[1]
REF = sys.argv[2]
SEED = 42


def set_seed(s: int) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(
        a.to(torch.float64) - b.to(torch.float64)))).item()


src = Path(TRIAL).read_text(encoding="utf-8")
ref_mod = load(REF, "refmod")
device = torch.device("cuda:0")

set_seed(SEED)
init = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_init_inputs()]
with torch.no_grad():
    set_seed(SEED)
    ref_model = ref_mod.Model(*init)
    set_seed(SEED)
    golden_model = ref_mod.Model(*[
        x.double() if torch.is_tensor(x) and x.is_floating_point() else x for x in init
    ]).to(device=device, dtype=torch.float64)
    set_seed(SEED)
    inputs = [x.to(device) if torch.is_tensor(x) else x for x in ref_mod.get_inputs()]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    out_tf32_ref = ref_model.to(device)(*inputs)
    torch.cuda.synchronize()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out_ieee_ref = ref_model.to(device)(*inputs)
    torch.cuda.synchronize()
    golden = golden_model(*[x.double() if torch.is_tensor(x) else x for x in inputs])

r_tf32 = rmse(golden, out_tf32_ref)
r_ieee = rmse(golden, out_ieee_ref)
print(f"reference RMSE vs fp64: tf32 {r_tf32:.6e}   ieee {r_ieee:.6e}\n")

outs = {}
tmp = Path("/tmp/dtype_probe")
tmp.mkdir(exist_ok=True)
for dt in ("fp16", "bf16", "tf32", "ieee"):
    variant = re.sub(r"('COMPUTE_DTYPE':\s*)'[a-z0-9]+'", rf"\1'{dt}'", src, count=1)
    assert f"'COMPUTE_DTYPE': '{dt}'" in variant, f"substitution failed for {dt}"
    p = tmp / f"v_{dt}.py"
    p.write_text(variant, encoding="utf-8")
    mod = load(str(p), f"v_{dt}")
    with torch.no_grad():
        set_seed(SEED)
        m = mod.ModelNew(*init)
        m.load_state_dict(ref_model.state_dict(), strict=False)
        m = m.to(device=device, dtype=torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        set_seed(SEED)
        o = m(*inputs)
        torch.cuda.synchronize()
    outs[dt] = o
    print(f"COMPUTE_DTYPE={dt:5s}  RMSE vs fp64 {rmse(golden, o):.6e}  "
          f"ratio to tf32-ref {rmse(golden, o) / r_tf32:.4f}")

print("\npairwise: are the four variants actually different tensors?")
keys = list(outs)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        same = torch.equal(outs[a], outs[b])
        d = rmse(outs[a], outs[b])
        flag = "  <-- BIT-IDENTICAL" if same else ""
        print(f"   {a:5s} vs {b:5s}: rmse {d:.6e}  identical={same}{flag}")

print("\nIf fp16 and tf32 are bit-identical the knob is inert on this candidate.")
