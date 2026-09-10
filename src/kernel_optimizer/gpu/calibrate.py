"""Host side of self-calibration: run the measurement, derive the thresholds, cache the result.

Split from `evaluation/calibration.py` deliberately. That module is pure -- models plus
`derive_thresholds` -- so the threshold logic is testable with no GPU and no worker. This one is
the part that needs a device, and it is thin: run the job, assemble, derive, flag, save.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from kernel_optimizer.evaluation.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    Calibration,
    Yardstick,
    cache_path,
    derive_thresholds,
    flag_suspect,
    load_cached,
    save,
)
from kernel_optimizer.gpu.jobs import make_calibrate_job


def calibration_from_worker(result: dict) -> Calibration:
    """Assemble a Calibration from a raw worker result, deriving the thresholds.

    Pure apart from reading the dict, so a recorded worker result replays into exactly the same
    calibration -- which is what makes a cached calibration auditable after the fact.
    """
    yardsticks = [
        Yardstick(
            name=y["name"], truth=y["truth"], gpu_ms=y["gpu_ms"],
            cpu_issue_ms=y["cpu_issue_ms"], flop_count=y["flop_count"],
            byte_count=y["byte_count"],
        )
        for y in result.get("yardsticks", [])
    ]
    dram_tbs = float(result.get("dram_tbs", 0.0) or 0.0)
    fp32_tflops = float(result.get("fp32_tflops", 0.0) or 0.0)
    spec_dram = float(result.get("spec_dram_tbs", 0.0) or 0.0)

    return Calibration(
        # Stamp the measurement set that produced this file. `load_cached` refuses anything
        # below the current constant, so forgetting this here would make every run
        # re-measure forever -- and hard-coding the integer instead of importing it would
        # let the two drift apart, which is the same silent-staleness bug one level up.
        schema_version=CALIBRATION_SCHEMA_VERSION,
        device_name=result.get("device_name", "unknown"),
        capability=list(result.get("capability", []) or []),
        sm_count=int(result.get("sm_count", 0) or 0),
        torch_version=str(result.get("torch_version", "") or ""),
        driver_version=str(result.get("driver_version", "") or ""),
        # G32: part of the cache identity. An older worker does not report it, and then this is ""
        # -- but such a cache is refused anyway on schema_version, which is the point of bumping it
        # alongside: "" here would otherwise be indistinguishable from a genuinely Triton-less box.
        triton_version=str(result.get("triton_version", "") or ""),
        dram_tbs=dram_tbs,
        fp32_tflops=fp32_tflops,
        tf32_tflops=float(result.get("tf32_tflops", 0.0) or 0.0),
        fp16_tflops=float(result.get("fp16_tflops", 0.0) or 0.0),
        bf16_tflops=float(result.get("bf16_tflops", 0.0) or 0.0),
        # G10: the Triton-reachable counterparts. Absent on an old worker, and then each is 0.0 and
        # the cuBLAS figure stands alone -- the behaviour before these were measured.
        fp32_triton_tflops=float(result.get("fp32_triton_tflops", 0.0) or 0.0),
        tf32_triton_tflops=float(result.get("tf32_triton_tflops", 0.0) or 0.0),
        fp16_triton_tflops=float(result.get("fp16_triton_tflops", 0.0) or 0.0),
        bf16_triton_tflops=float(result.get("bf16_triton_tflops", 0.0) or 0.0),
        empty_launch_floor_ms=float(result.get("empty_launch_floor_ms", 0.0) or 0.0),
        spec_dram_tbs=spec_dram,
        l2_bytes=int(result.get("l2_bytes", 0) or 0),
        yardsticks=yardsticks,
        thresholds=derive_thresholds(yardsticks, dram_tbs, fp32_tflops),
        tiers=dict(result.get("tiers") or {}),
        suspect=flag_suspect(dram_tbs, spec_dram),
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def ensure_calibration(
    worker,
    run_root: Path,
    *,
    recalibrate: bool = False,
    timeout_s: float = 900.0,
    identity_hint: str | None = None,
    store=None,
) -> Calibration | None:
    """Return this box's calibration, measuring it only when the cache cannot serve.

    Cached beside the runs rather than inside one: it describes the BOX, not the run, and
    re-measuring costs a couple of minutes of exclusive GPU time that every run would pay.

    The cache is keyed on device identity, so it is never silently reused across a hardware or
    driver change -- every classification is a fraction of these ceilings, so a stale one
    produces confident verdicts computed against a different card's limits.

    Returns None (never raises) when the measurement fails: a box without a working calibration
    must still be able to run, with the classifier reporting `unknown` rather than the harness
    refusing to start.
    """
    path = cache_path(run_root)
    if not recalibrate:
        cached = load_cached(path, identity_hint)
        if cached is not None:
            if store is not None:
                # The SAME fields a fresh measurement journals. An earlier version omitted
                # `tiers`, `tf32_tflops`, `empty_launch_floor_ms` and `thresholds` here, so a run
                # that hit the cache -- which is the normal case, since calibration is per-box --
                # produced a report with no tier line and no thresholds, while a run that
                # re-measured produced a full one. The data was in the cache all along; only the
                # event was thin, and the report reads the event. Observed live on
                # run-l1-42-20260908-015408: tier1 read as None in the log while the cache file
                # held tier1_sass_and_occupancy: true.
                store.append("CALIBRATION_LOADED",
                             {"source": "cache", "path": str(path),
                              "device": cached.device_name,
                              "dram_tbs": cached.dram_tbs,
                              "fp32_tflops": cached.fp32_tflops,
                              "tf32_tflops": cached.tf32_tflops,
                              "fp16_tflops": cached.fp16_tflops,
                              "bf16_tflops": cached.bf16_tflops,
                              "empty_launch_floor_ms": cached.empty_launch_floor_ms,
                              "ridge_flop_per_byte": cached.ridge_flop_per_byte,
                              "thresholds": (cached.thresholds.model_dump()
                                             if cached.thresholds else None),
                              "tiers": cached.tiers,
                              "measured_at": cached.measured_at,
                              "suspect": cached.suspect})
            return cached

    # Exclusive: this measures ceilings, so a neighbour job would depress every number and the
    # whole point is that these are what a kernel can actually get on an otherwise idle box.
    result = worker.run_job(make_calibrate_job(), timeout_s=timeout_s, tag="calibrate",
                            lock_mode="exclusive")
    if not result.get("ok"):
        if store is not None:
            store.append("CALIBRATION_FAILED",
                         {"failure_kind": result.get("failure_kind"),
                          "log_tail": (result.get("log_tail") or "")[:1000]})
        return None

    cal = calibration_from_worker(result)
    save(path, cal)
    if store is not None:
        store.append("CALIBRATION_MEASURED", {
            "path": str(path), "device": cal.device_name,
            "dram_tbs": cal.dram_tbs, "fp32_tflops": cal.fp32_tflops,
            "tf32_tflops": cal.tf32_tflops,
            "fp16_tflops": cal.fp16_tflops,
            "bf16_tflops": cal.bf16_tflops,
            "empty_launch_floor_ms": cal.empty_launch_floor_ms,
            "ridge_flop_per_byte": cal.ridge_flop_per_byte,
            "thresholds": cal.thresholds.model_dump() if cal.thresholds else None,
            "tiers": cal.tiers,
            "suspect": cal.suspect,
        })
    return cal
