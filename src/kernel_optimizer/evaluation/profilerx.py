"""Light profiler: map worker triton metadata into ProfileRecord. Never fabricates."""

from __future__ import annotations

from typing import Any

from kernel_optimizer.models.core import ProfileRecord


class LightProfiler:
    def extract(self, worker_result: dict[str, Any]) -> ProfileRecord:
        triton = worker_result.get("triton")
        if not triton or not triton.get("kernels"):
            return ProfileRecord(compile_s=(triton or {}).get("compile_s"))
        kernels = triton["kernels"]
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
            compile_s=triton.get("compile_s"),
            kernel_names=[k["name"] for k in kernels if k.get("name")],
        )
