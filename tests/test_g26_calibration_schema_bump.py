"""A newly load-bearing calibration field must not be served as 0 from an old cache.

THE FAILURE THIS PREVENTS, caught live on 2026-09-10. G26 made `l2_bytes` load-bearing: it is the
precondition deciding whether the DRAM dimension applies at all. But the field had been *measured*
and cached for a while with nothing reading it, so the A800's cached calibration held no l2-related
key whatsoever. Loading it would give `l2_bytes = 0`, the working-set gate would never fire, and
every verdict would still look like a verdict -- the same silent-zero shape as G10's ceilings one
schema version earlier, and the reason CALIBRATION_SCHEMA_VERSION exists.

The rule the module's own docstring states: bump the version whenever a new quantity starts being
measured. The gap this file closes is that nothing enforced it, so a field could become load-bearing
without a bump -- which is exactly what happened.
"""
from __future__ import annotations

import inspect

from kernel_optimizer.evaluation import calibration as cal_mod
from kernel_optimizer.evaluation.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    Calibration,
)


def test_a_cache_below_the_current_schema_is_refused():
    """The mechanism itself: an older cache must be re-measured, not loaded with zeros."""
    src = inspect.getsource(cal_mod)
    assert "cal.schema_version < CALIBRATION_SCHEMA_VERSION" in src, (
        "nothing refuses a stale cache any more, so a newly added measurement is served as its "
        "permissive default (0.0 / 0) indefinitely and silently")


def test_the_version_covers_l2_bytes():
    """G26's field is in the measurement set, so pre-G26 caches are invalidated.

    Asserted as a number rather than a string search: the point is that the version was BUMPED when
    l2_bytes became load-bearing, and only a number can carry that.
    """
    assert CALIBRATION_SCHEMA_VERSION >= 4, (
        "l2_bytes became the DRAM dimension's applicability precondition (G26) without bumping the "
        "calibration schema version, so every box with an older cache reads l2_bytes = 0, the "
        "working-set gate never fires, and the L2-resident mislabelling silently persists")


def test_l2_bytes_defaults_to_zero_and_zero_means_unmeasured():
    """The default must be the SAFE direction: unmeasured disables the gate, never asserts it."""
    c = Calibration(device_name="x", capability=[8, 0], sm_count=108,
                    schema_version=CALIBRATION_SCHEMA_VERSION,
                    dram_tbs=1.0, fp32_tflops=1.0)
    assert c.l2_bytes == 0, (
        "l2_bytes has a non-zero default, so a box that never measured it would get a fabricated "
        "L2 size and could declare a real bandwidth verdict inapplicable")


def test_every_measured_ceiling_is_listed_in_the_version_log():
    """The version comment must name what each bump added, or the next person cannot tell.

    Weak by nature (it reads a comment), but its absence is what let l2_bytes drift: the log is the
    only place recording WHICH fields a version covers.
    """
    src = inspect.getsource(cal_mod)
    log_start = src.index("#   1  dram/fp32")
    log = src[log_start:log_start + 600]
    for token in ("fp16_tflops", "triton_tflops", "l2_bytes"):
        assert token in log, (
            "%s is measured but not recorded in the schema-version log, so a future reader cannot "
            "tell which caches contain it: %s" % (token, log))
