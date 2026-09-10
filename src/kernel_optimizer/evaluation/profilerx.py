"""Light profiler: worker metadata -> ProfileRecord, for every backend. Never fabricates.

BACKEND-NEUTRAL BY DESIGN. It used to read only Triton's compiled-kernel object, so a
`backend: "cuda"` candidate produced an empty ProfileRecord: no registers, no spills, no shared
memory. That is not a property of the backend -- it was a shortcut in this file -- and it
degraded the paper's own feedback loop (tuning evidence -> bottleneck report -> structural
rewrite) on the backend with the HIGHER expressiveness ceiling, which is backwards.

CUDA C++, CUTLASS and CuTe all compile through nvcc to a cubin, so ONE reader covers all three
(`_extract_cubin_metadata` in the worker, via `cuobjdump -res-usage`). This file just maps
whichever source is present into the same record, so every downstream consumer -- the tuning
stats' resource-saturation logic, the analyst's inputs, trials.csv -- works identically
regardless of how the candidate was written.
"""

from __future__ import annotations

from typing import Any

from kernel_optimizer.models.core import ProfileRecord


class LightProfiler:
    def extract(self, worker_result: dict[str, Any]) -> ProfileRecord:
        triton = worker_result.get("triton")
        cubin = worker_result.get("cubin")
        # Launch overhead is backend-independent (it is a property of the CALL, not of how the
        # kernel was written), so it is read outside the triton/cubin branch and survives even
        # when no resource metadata could be collected at all.
        over = worker_result.get("launch_overhead") or {}
        overhead_fields = {
            "cpu_issue_ms": over.get("cpu_issue_ms"),
            "overhead_gpu_ms": over.get("gpu_ms"),
            "wall_ms": over.get("wall_ms"),
        }
        # Per-candidate cost (G4/G6). Like launch overhead this is a property of RUNNING the
        # candidate rather than of how it compiled, so it is read outside the triton/cubin branch
        # and survives a candidate whose resource metadata could not be collected at all. The
        # worker only produces it on the Triton path today (it needs the module object), so a
        # cubin-only candidate carries the note rather than silent None -- absent and zero are
        # different answers, and only the note distinguishes them.
        cost = (triton or {}).get("candidate_cost") or {}
        cost_fields = {
            "peak_alloc_bytes": cost.get("peak_alloc_bytes"),
            "peak_reserved_bytes": cost.get("peak_reserved_bytes"),
            "peak_above_resident_bytes": cost.get("peak_above_resident_bytes"),
            "candidate_aten_bytes": cost.get("candidate_aten_bytes"),
            "candidate_aten_ops": cost.get("candidate_aten_ops"),
            "threads_launched": cost.get("threads_launched"),
            "launches": cost.get("launches") or [],
            "cost_notes": cost.get("cost_notes") or [],
        }
        # Triton first: when both are present (a candidate mixing a jit kernel with an inline
        # CUDA helper) the Triton record carries num_warps/num_stages, which a cubin cannot --
        # those are properties of the LAUNCH, not of the compiled code.
        source = triton if (triton and triton.get("kernels")) else cubin
        if not source or not source.get("kernels"):
            return ProfileRecord(compile_s=(triton or {}).get("compile_s"),
                                 **overhead_fields, **cost_fields)
        kernels = source["kernels"]

        # Aggregate across kernels in the launch: max regs/shared is the binding value.
        def _agg(key: str) -> int | None:
            vals = [k[key] for k in kernels if k.get(key) is not None]
            return max(vals) if vals else None

        # Tier 1 statics (step 5). Aggregated across the launch's kernels the way the resource
        # figures are, but with the aggregation matched to what each number MEANS:
        #   sass       summed -- the instruction mix of the whole launch is the sum of its parts,
        #              and "does this launch use tensor cores at all" is then just > 0.
        #   occupancy  taken from the WORST kernel, because the launch is limited by its least
        #              occupant. Averaging would hide a kernel stuck at 17% behind one at 100%.
        sass_total: dict | None = None
        sass_rows = [k["sass"] for k in kernels if k.get("sass")]
        if sass_rows:
            keys = {key for row in sass_rows for key in row}
            sass_total = {key: sum(int(row.get(key) or 0) for row in sass_rows) for key in keys}
        occ_rows = [k["occupancy"] for k in kernels if k.get("occupancy")]
        worst_occ = min(occ_rows, key=lambda o: o.get("occupancy", 1.0)) if occ_rows else None
        notes: list[str] = []
        for k in kernels:
            notes.extend(k.get("statics_notes") or [])
            if k.get("statics_note"):
                notes.append(k["statics_note"])

        return ProfileRecord(
            n_regs=_agg("n_regs"),
            n_spills=_agg("n_spills"),
            shared_bytes=_agg("shared"),
            num_warps=_agg("num_warps"),
            num_stages=_agg("num_stages"),
            compile_s=(triton or {}).get("compile_s"),
            kernel_names=[k["name"] for k in kernels if k.get("name")],
            # Which backend the numbers came from, and -- for a cubin -- whether the launched
            # kernels could be identified. `no_name_match` means the resources are the union
            # over the whole cubin, which for CUTLASS may include template variants that never
            # ran, so a reader must not treat them as measured for the executing kernel.
            profile_source=("triton" if source is triton else "cubin"),
            launched_filter=source.get("launched_filter"),
            sass=sass_total,
            occupancy=worst_occ,
            statics_notes=sorted(set(notes)),
            **overhead_fields,
            **cost_fields,
        )
