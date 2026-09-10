"""Measure the five per-candidate fields on ONE candidate and merge them into its profile.json.

Needed because the G9 `rich` arm's entire content is the per-candidate measurements added for
G4/G6/G2, and a profile extracted from a run that predates them carries none of them. The A/B
correctly REFUSES to run in that state -- otherwise `rich` is a silent duplicate of `verdict` and two
identical arms get reported as a comparison.

Reuses `probe_per_candidate_cost.measure_one` rather than re-implementing the measurement, so the
numbers here are the same quantity that probe validated (peak alloc varies 9.6-41.3% between
candidates; aten-level traffic is a 15-82% lower bound; candidate-level FLOPs are NOT measurable --
FlopCounterMode reads 0 through a fused Triton kernel, correct but useless).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the task's reference problem file")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--profile-json", required=True, help="profile to merge into (updated in place)")
    args = ap.parse_args()

    import torch

    from probe_per_candidate_cost import load_module, measure_one

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    ref_ctx: dict = {}
    exec(compile(io.open(args.ref, encoding="utf-8").read(), "<ref>", "exec"), ref_ctx)  # noqa: S102

    mod = load_module(args.candidate, 0)
    m = measure_one(mod, ref_ctx, device, torch)

    threads = sum(l.get("threads") or 0 for l in m.get("launches_warm", []))
    fields = {
        "peak_alloc_bytes": int(m["peak_alloc_bytes"]),
        "peak_reserved_bytes": int(m.get("peak_reserved_bytes") or 0) or None,
        "candidate_aten_bytes": int(m["aten_bytes"]),
        "candidate_aten_ops": int(m["aten_ops"]),
        "threads_launched": int(threads) or None,
    }
    print("MEASURED on this candidate")
    for k, v in fields.items():
        if k.endswith("_bytes") and v:
            print("  %-24s %s  (%.1f MiB)" % (k, v, v / 2**20))
        else:
            print("  %-24s %s" % (k, v))

    missing = [k for k, v in fields.items() if v is None]
    if missing:
        # Say it plainly rather than writing nulls that make the arm partly hollow.
        print()
        print("NOTE: %s could not be measured on this candidate. The `rich` arm will carry the "
              "fields that WERE measured; that is still a strict superset of `verdict`." % missing)

    path = Path(args.profile_json)
    prof = json.loads(path.read_text(encoding="utf-8"))
    prof.update({k: v for k, v in fields.items() if v is not None})
    path.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    print()
    print("merged into %s" % path)
    present = [k for k in fields if prof.get(k) is not None]
    print("rich arm will now add: %s" % ", ".join(present))
    return 0 if present else 1


if __name__ == "__main__":
    raise SystemExit(main())
