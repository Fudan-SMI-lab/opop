"""Can we measure PER-CANDIDATE traffic, arithmetic and peak memory? A probe, not a design.

WHY THIS EXISTS

`bottleneck.py` divides by `byte_count` and `flop_count`, and the orchestrator passes
`cost.compulsory_bytes` / `cost.flop_count` -- both measured ONCE on the reference and shared by
every candidate of a task. Verified on disk across 5 runs: gpu_ms varies up to 4.6x between
candidates while the derived byte_count varies <=0.36% and flop_count <=0.008% (that residual is
just rounding in the stored 4-decimal fields). So the two "pressure" dimensions are 1/latency.

The constant is not a bug in itself -- it is a deliberate task-level yardstick, and for the
question "can this task ever be compute-bound on this card" it is the RIGHT number, because it is
the floor no implementation can go below. The defect is using a task-level floor as if it were a
per-candidate measurement.

So: what CAN be measured per candidate, without hardware counters (ncu is permanently blocked by
ERR_NVGPUCTRPERM in our container)? This probe tries four things on real candidate files and
reports which of them (a) vary between candidates, (b) cost little enough to afford, (c) agree
with an independently known answer.

  M1  peak device memory      torch.cuda.max_memory_allocated() around the forward
  M2  materialized traffic    a __torch_dispatch__ counter around the CANDIDATE (the same
                              mechanism task_cost.py uses on the reference) -- catches every
                              aten-level tensor the candidate creates, but is BLIND inside a
                              fused Triton kernel, which is exactly where our candidates do
                              their work. Expected to under-report, and the size of the
                              under-report is itself the interesting number.
  M3  launch geometry         grid size x block size per kernel, captured by wrapping
                              JITFunction.run. Gives threads launched, which is the missing
                              factor that turns nvdisasm's per-thread static instruction counts
                              into a whole-kernel logical access count.
  M4  logical traffic         M3's thread count x the kernel's static global-access instruction
                              counts from the cubin (already extracted as sass.global_load /
                              global_store). This is a LOGICAL upper bound: it cannot see
                              predicated-off lanes, and it counts an L2 hit as traffic.

A positive control is mandatory (see probe_needs_a_positive_control): each measurement must
VARY across candidates. A number that is constant across structurally different candidates is
either a broken accessor or another task-level constant in disguise -- and a constant is the
failure mode that looks like a result.

Usage:  probe_per_candidate_cost.py <ref.py> <cand1.py> [cand2.py ...]
"""
from __future__ import annotations

import json
import sys
import time
import traceback


def load_module(src_path: str, index: int):
    """Import a candidate file under a unique module name.

    The unique name matters: reusing one name lets Python's import machinery hand back the
    first file's module object, so every later candidate silently reports the first one's
    numbers -- a probe that looks like it works while answering the wrong question.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"kopt_cost_probe_{index}", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_one(module, ref_ctx, device, torch):
    """All four measurements in one forward pass each, so the cost is one launch, not four."""
    from torch.utils._python_dispatch import TorchDispatchMode

    get_inputs = ref_ctx["get_inputs"]
    get_init_inputs = ref_ctx.get("get_init_inputs", lambda: [])

    out: dict = {}

    # ---- M3: launch geometry. Wrap JITFunction.run so every launch records its grid.
    launches: list[dict] = []
    from triton.runtime.jit import JITFunction
    original_run = JITFunction.run

    def run_recording(self, *args, grid=None, warmup=False, **kwargs):
        kernel = original_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        try:
            g = grid
            if callable(g):
                # A grid lambda takes the meta dict; Triton has already resolved it internally,
                # but we cannot see that value, so re-resolve with what we can reconstruct.
                g = None
            n_blocks = None
            if isinstance(g, tuple):
                n_blocks = 1
                for d in g:
                    n_blocks *= int(d)
            elif isinstance(g, int):
                n_blocks = int(g)
            meta = getattr(kernel, "metadata", None)
            nw = int(getattr(meta, "num_warps", 0) or 0)
            launches.append({
                "kernel": getattr(self, "__name__", "?"),
                "n_blocks": n_blocks,
                "num_warps": nw,
                "threads_per_block": nw * 32 if nw else None,
                "threads": (n_blocks * nw * 32) if (n_blocks and nw) else None,
                "grid_was_callable": callable(grid),
            })
        except Exception:  # noqa: BLE001 -- one unreadable launch must not lose the rest
            launches.append({"kernel": getattr(self, "__name__", "?"), "n_blocks": None})
        return kernel

    # ---- M2: materialized traffic at the aten level, on the CANDIDATE this time.
    class ByteCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.bytes = 0
            self.ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            res = func(*args, **kwargs)
            self.ops += 1
            seen_here = set()

            def acc(t):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    return 0
                key = id(t)
                if key in seen_here:
                    return 0
                seen_here.add(key)
                return t.numel() * t.element_size()

            def walk(x):
                if isinstance(x, torch.Tensor):
                    self.bytes += acc(x)
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        walk(y)

            walk(args)
            walk(kwargs.values() if kwargs else ())
            walk(res)
            return res

    with torch.no_grad():
        init_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x
                       for x in get_init_inputs()]
        model = module.ModelNew(*init_inputs).to(device)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

        # warm up first: compilation allocates, and we do not want compile-time allocation in
        # the peak-memory number.
        JITFunction.run = run_recording
        try:
            model(*inputs)
            torch.cuda.synchronize(device)
        finally:
            JITFunction.run = original_run
        out["launches_warm"] = list(launches)

        # ---- M1: peak device memory of the forward alone.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before_alloc = torch.cuda.memory_allocated(device)
        t0 = time.perf_counter()
        model(*inputs)
        torch.cuda.synchronize(device)
        fwd_s = time.perf_counter() - t0
        out["peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated(device))
        out["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        out["alloc_before_fwd"] = int(before_alloc)
        out["peak_above_resident"] = out["peak_alloc_bytes"] - int(before_alloc)
        out["fwd_s"] = round(fwd_s, 6)

        # ---- M2, on a separate pass: the dispatch mode changes timing, so never combine it
        # with the peak-memory pass.
        bc = ByteCounter()
        launches.clear()
        with bc:
            model(*inputs)
            torch.cuda.synchronize(device)
        out["aten_bytes"] = int(bc.bytes)
        out["aten_ops"] = int(bc.ops)

    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    import torch
    ref_path, cand_paths = sys.argv[1], sys.argv[2:]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ref_ctx: dict = {}
    exec(compile(open(ref_path, encoding="utf-8").read(), "<ref>", "exec"), ref_ctx)  # noqa: S102

    rows = []
    for i, p in enumerate(cand_paths):
        row = {"path": p}
        try:
            t0 = time.perf_counter()
            mod = load_module(p, i)
            row.update(measure_one(mod, ref_ctx, device, torch))
            row["probe_s"] = round(time.perf_counter() - t0, 3)
        except Exception as exc:  # noqa: BLE001 -- one bad candidate must not lose the batch
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["trace"] = traceback.format_exc()[-600:]
        rows.append(row)
        print("  probed %-44s %s" % (p.split("/")[-1][:44],
                                     row.get("error") or "ok %.1fs" % row.get("probe_s", 0)))

    print()
    print("=" * 108)
    print("PER-CANDIDATE COST -- does each measurement VARY between candidates?")
    print("=" * 108)
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("  every candidate failed -- nothing measured. VERDICT: INCONCLUSIVE")
        for r in rows:
            print("   ", r["path"], r.get("error"))
        return 2

    print("%-40s %14s %14s %12s %10s %8s"
          % ("candidate", "peak_alloc MiB", "aten_bytes MiB", "aten_ops", "threads", "fwd_ms"))
    for r in ok:
        thr = sum(l.get("threads") or 0 for l in r.get("launches_warm", []))
        print("%-40s %14.2f %14.2f %12d %10s %8.2f"
              % (r["path"].split("/")[-1][:40],
                 r["peak_alloc_bytes"] / 2**20, r["aten_bytes"] / 2**20, r["aten_ops"],
                 thr or "-", r["fwd_s"] * 1e3))

    print()
    for field, label, unit in (("peak_alloc_bytes", "M1 peak device memory", 2**20),
                               ("aten_bytes", "M2 aten-level traffic", 2**20)):
        vals = [r[field] for r in ok]
        lo, hi = min(vals), max(vals)
        spread = 100 * (hi - lo) / hi if hi else 0.0
        verdict = ("VARIES -- usable as a per-candidate dimension" if spread > 1.0
                   else "CONSTANT across candidates -- either broken or another task-level "
                        "constant. NOT usable.")
        print("  %-26s %.2f - %.2f (spread %.2f%%)   %s"
              % (label, lo / unit, hi / unit, spread, verdict))

    thr = [sum(l.get("threads") or 0 for l in r.get("launches_warm", [])) for r in ok]
    n_callable = sum(1 for r in ok for l in r.get("launches_warm", []) if l.get("grid_was_callable"))
    n_launch = sum(len(r.get("launches_warm", [])) for r in ok)
    print("  %-26s %s" % ("M3 launch geometry",
                          "%d launches seen, %d had a CALLABLE grid (thread count unavailable "
                          "for those)" % (n_launch, n_callable)))
    if thr and max(thr) > 0:
        lo, hi = min(t for t in thr if t), max(thr)
        print("  %-26s %d - %d threads (spread %.1f%%)"
              % ("", lo, hi, 100 * (hi - lo) / hi))
    if n_callable == n_launch and n_launch:
        print("      *** every grid was a lambda, so M3 measured NOTHING. A Triton grid is")
        print("          normally a lambda over meta, so this is the expected case and the")
        print("          wrapper must resolve it, not read `grid` -- see the docstring.")

    print()
    print("  cost per candidate: %.1f - %.1f s"
          % (min(r["probe_s"] for r in ok), max(r["probe_s"] for r in ok)))
    json.dump(rows, open("per_candidate_cost.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("  full detail -> per_candidate_cost.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
