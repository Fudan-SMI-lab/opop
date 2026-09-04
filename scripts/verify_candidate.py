"""Independently verify a published candidate's correctness and speedup.

The L3:48 candidates were rejected by the old (overflowing) cosine gate and are now
accepted, so the first one through deserves an out-of-band check rather than trusting the
gate that just changed. This does not reuse worker_main's gate at all: it recomputes the
reference in float64 (the trustworthy value), scores the candidate against it, and times
both with CUDA events in the same process.

A candidate that is correct here AND shows a speedup close to the harness's number is
genuinely fast. A candidate that is correct but much slower here than the harness claimed
would indicate a timing artifact.

Usage: ... python scripts/verify_candidate.py <run_dir> <candidate_id> [params.json]
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def time_module(fn, inputs, n: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        starts[i].record()
        fn(*inputs)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def main() -> int:
    run_dir = Path(sys.argv[1])
    cand_id = sys.argv[2]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ref_path = manifest["task"]["ref_path"].replace("\\", "/")
    if ref_path[1:3] == ":/":  # D:/... -> /mnt/d/...
        ref_path = f"/mnt/{ref_path[0].lower()}{ref_path[2:]}"

    cand_dir = run_dir / "candidates" / cand_id
    src = next((p for p in (cand_dir / "source.py", cand_dir / "witness_default.py")
                if p.exists()), None)
    if src is None:
        print(f"no source found under {cand_dir}")
        return 2
    print(f"reference: {ref_path}\ncandidate: {src}\n")

    ref_mod = load(ref_path, "refmod")
    cand_mod = load(src, "candmod")

    torch.manual_seed(0)
    dev = "cuda"
    init = ref_mod.get_init_inputs()
    ref_model = ref_mod.Model(*init).to(dev)
    inputs = [t.to(dev) if torch.is_tensor(t) else t for t in ref_mod.get_inputs()]

    torch.manual_seed(0)
    cand_model = cand_mod.ModelNew(*init).to(dev)
    # Copy any shared parameters so both models see identical weights.
    ref_state = ref_model.state_dict()
    matched = [k for k in cand_model.state_dict() if k in ref_state]
    if matched:
        cand_model.load_state_dict({**cand_model.state_dict(),
                                    **{k: ref_state[k] for k in matched}})
        print(f"copied {len(matched)} shared parameter tensors into the candidate")

    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        y_ref = ref_model(*inputs)
        y_ref = y_ref[0] if isinstance(y_ref, tuple) else y_ref
        y64 = ref_model.double()(*[t.double() if torch.is_tensor(t) else t for t in inputs])
        y64 = y64[0] if isinstance(y64, tuple) else y64
        ref_model.float()
        y_cand = cand_model(*inputs)
        y_cand = y_cand[0] if isinstance(y_cand, tuple) else y_cand

    def score(a, b, label):
        a32, b32 = a.float(), b.float()
        rel = (a32 - b32).abs() / (a32.abs() + 1e-7)
        x, y = a32.flatten().double(), b32.flatten().double()
        sc = max(x.abs().max().item(), y.abs().max().item()) or 1.0
        cos = torch.dot(x / sc, y / sc).item() / ((x / sc).norm() * (y / sc)).norm().item()
        print(f"  {label:28s} frac_within_1%={rel.lt(0.01).float().mean().item():.6f} "
              f"cos={cos:.8f} med_rel={rel.median().item():.2e}")

    print("\ncorrectness (independent of the harness gate):")
    score(y64.float(), y_ref, "fp64 truth vs ref fp32")
    score(y64.float(), y_cand, "fp64 truth vs CANDIDATE")
    score(y_ref, y_cand, "ref fp32 vs CANDIDATE")

    with torch.no_grad():
        t_ref = time_module(ref_model, inputs)
        t_cand = time_module(cand_model, inputs)
    print(f"\ntiming (median of 50, CUDA events, same process):")
    print(f"  reference eager : {t_ref:.2f} ms")
    print(f"  candidate       : {t_cand:.2f} ms")
    print(f"  speedup vs eager: {t_ref / t_cand:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
