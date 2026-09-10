"""Triton's version must be part of what a cached calibration may be reused on (G32).

WHY THIS EXISTS. Four of the measured ceilings are `*_triton_tflops` (G10) -- what a Triton kernel
can actually reach on this box -- and every candidate this harness produces is written in Triton. So
Triton IS the code generator, and its version can move an achievable ceiling with no change to the
card, the driver, or torch. The two boxes in use run 3.5.1 and 3.4.0, and the setup record states
those generate different code: different register counts, different shared usage, and sometimes the
difference between a tile compiling and not.

`Calibration.identity()` decides when a cached calibration may be reused. Before G32 it carried
device name, capability, SM count, torch and driver -- but not Triton. A lone `pip install -U triton`
on one box therefore kept serving ceilings measured by the OLD compiler, silently, and every verdict
downstream is a fraction of those ceilings.

This is the same class of defect as G10's and G26's: nothing raises, and a stale ceiling still reads
exactly like a ceiling. It is also the direct code form of the user's ruling that a toolchain version
is part of the environment -- not a dimension to model, but a reason a conclusion must not be
carried across.

WHAT EACH TEST DRIVES. `identity()` and `load_cached` are called for real; nothing here re-implements
either. The negative controls matter as much as the positive ones -- an identity that changes when
nothing changed would re-measure on every single run, which costs exclusive GPU time per run and
would be "fixed" by reverting this.
"""
from __future__ import annotations

import json

from kernel_optimizer.evaluation.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    Calibration,
    load_cached,
)


def _cal(**over) -> Calibration:
    """A calibration with every identity field set, so a dropped field shows up as a collision."""
    base = dict(device_name="NVIDIA A800 80GB PCIe", capability=[8, 0], sm_count=108,
                torch_version="2.8.0+cu128", driver_version="12.8", triton_version="3.4.0",
                schema_version=CALIBRATION_SCHEMA_VERSION, dram_tbs=1.686, fp32_tflops=19.0)
    base.update(over)
    return Calibration(**base)


def test_triton_version_changes_the_identity():
    """The whole point: two calibrations differing ONLY in Triton are not interchangeable."""
    a = _cal(triton_version="3.4.0")
    b = _cal(triton_version="3.5.1")
    assert a.identity() != b.identity(), (
        "identity() is blind to the Triton version, so upgrading Triton on a box keeps serving "
        "ceilings measured by the previous compiler -- including the four *_triton_tflops figures, "
        "which are BY DEFINITION properties of that compiler")


def test_identity_is_stable_when_nothing_changed():
    """Negative control. An identity that moves on its own re-measures every run.

    Without this, "make identity more sensitive" could be satisfied by putting a timestamp in it,
    which would pass the test above and cost exclusive GPU time on every single run.
    """
    assert _cal().identity() == _cal().identity(), (
        "identity() is not a pure function of the recorded fields, so no cache can ever be reused")


def test_the_other_identity_fields_still_count():
    """Negative control on the opposite side: adding Triton must not have replaced anything."""
    base = _cal()
    for field, other in (("device_name", "NVIDIA GeForce RTX 4090"),
                         ("capability", [8, 9]),
                         ("sm_count", 128),
                         ("torch_version", "2.9.1+cu129"),
                         ("driver_version", "12.9")):
        assert _cal(**{field: other}).identity() != base.identity(), (
            "%s dropped out of identity(), so a calibration would be reused across a change that "
            "moves the ceilings" % field)


def test_a_pre_g32_cache_is_refused(tmp_path):
    """A cache written before this field existed must be re-measured, not loaded with "".

    This is the half that identity alone cannot do. Every pre-G32 cache has no `triton_version` key,
    so it validates with "" -- which is ALSO the honest value for a box with no Triton installed. The
    two are indistinguishable from the file, and on the common call path `identity_hint` is None so
    identity is never compared at all. Only the schema bump refuses them.
    """
    stale = _cal().model_dump()
    stale["schema_version"] = 4          # the version before G32
    stale.pop("triton_version")          # as an actual pre-G32 file looks
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(stale), encoding="utf-8")

    assert load_cached(path, None) is None, (
        "a calibration predating triton_version was loaded, so its Triton-measured ceilings are "
        "being reused with no record of which Triton produced them -- and with identity_hint None, "
        "identity() is not consulted, so the schema version is the ONLY thing that can refuse it")


def test_a_current_cache_still_loads(tmp_path):
    """Negative control on the bump: refusing everything would be a silent recalibrate-always."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_cal().model_dump()), encoding="utf-8")
    cal = load_cached(path, None)
    assert cal is not None, "a cache at the current schema version was refused"
    assert cal.triton_version == "3.4.0", "the field round-trips through the cache file as empty"


def test_the_worker_reports_the_field():
    """The field has to be MEASURED, not just modelled -- otherwise it is always "".

    Asserted through `calibration_from_worker`, the real assembly path, using a worker result dict
    rather than a live GPU. A model field with no producer is the dormant-defect shape that G26 hit:
    `DevicePeaks` had no `l2_bytes` at all, so the gate reading it could never fire.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker({
        "ok": True, "device_name": "NVIDIA A800 80GB PCIe", "capability": [8, 0], "sm_count": 108,
        "torch_version": "2.8.0+cu128", "driver_version": "12.8", "triton_version": "3.4.0",
        "dram_tbs": 1.686, "fp32_tflops": 19.0, "yardsticks": [],
    })
    assert cal.triton_version == "3.4.0", (
        "the worker's triton_version does not reach the Calibration, so identity() always sees "
        "the empty default and the G32 check is inert -- present in the model, absent in practice")
    assert cal.triton_version in cal.identity(), "the measured value is not in the identity string"


def test_a_worker_without_triton_yields_a_stable_empty():
    """A CUDA-only box must still calibrate, and its identity must not thrash."""
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    result = {"ok": True, "device_name": "x", "capability": [8, 0], "sm_count": 1,
              "torch_version": "2.8.0", "driver_version": "12.8",
              "dram_tbs": 1.0, "fp32_tflops": 1.0, "yardsticks": []}
    a = calibration_from_worker(result)
    b = calibration_from_worker(result)
    assert a.triton_version == "", "a missing triton_version became something other than empty"
    assert a.identity() == b.identity(), (
        "a Triton-less box gets a different identity on every calibration, so it re-measures every "
        "run -- the cost this field was meant to avoid paying twice")
