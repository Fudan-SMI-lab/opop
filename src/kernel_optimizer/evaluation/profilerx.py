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
        # Triton first: when both are present (a candidate mixing a jit kernel with an inline
        # CUDA helper) the Triton record carries num_warps/num_stages, which a cubin cannot --
        # those are properties of the LAUNCH, not of the compiled code.
        source = triton if (triton and triton.get("kernels")) else cubin
        if not source or not source.get("kernels"):
            return ProfileRecord(compile_s=(triton or {}).get("compile_s"), **overhead_fields)
        kernels = source["kernels"]

        # Aggregate across kernels in the launch: max regs/shared is the binding value.
        def _agg(key: str) -> int | None:
            vals = [k[key] for k in kernels if k.get(key) is not None]
            return max(vals) if vals else None

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
            **overhead_fields,
        )
