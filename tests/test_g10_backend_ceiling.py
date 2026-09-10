"""G10: the arithmetic ceiling must be one the candidate's own backend can reach.

Every candidate this harness generates is Triton; calibration measured its compute ceilings with
`torch.matmul`, i.e. cuBLAS. Measured on box 1 (RTX 4090) with every figure gated on correctness
against a fp64 reference, the two disagree in BOTH directions:

    prec   cuBLAS   Triton   ratio   rel err
    fp32    54.20    45.61   0.841   1.62e-06
    tf32    88.14    86.58   0.982   7.91e-04
    fp16   159.47   174.78   1.096   2.07e-04
    bf16   162.29   175.18   1.079   1.66e-03

Each test below asserts a behaviour and states what breaks without it.
"""
from __future__ import annotations

import inspect

from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify
from kernel_optimizer.evaluation.calibration import CALIBRATION_SCHEMA_VERSION, Calibration

# Box 1, correctness-gated.
BOX1 = dict(dram_tbs=0.911, fp32_tflops=54.20, tf32_tflops=88.14,
            fp16_tflops=159.47, bf16_tflops=162.29,
            fp32_triton_tflops=45.61, tf32_triton_tflops=86.58,
            fp16_triton_tflops=174.78, bf16_triton_tflops=175.18)


def test_a_ceiling_is_the_max_over_measured_backends_not_one_library():
    """cuBLAS alone is wrong in both directions, so neither library alone is the roof.

    At fp32, scoring a Triton candidate against cuBLAS invents 16% of headroom no tiling change can
    reach. At fp16, cuBLAS is an ANTI-ceiling: the fraction exceeds 100%, which reads downstream as
    "saturated, stop optimizing" for a kernel that is at no limit at all.
    """
    peaks = DevicePeaks(**BOX1)

    # fp16: Triton is higher, so Triton sets the ceiling.
    ceil_fp16, name_fp16 = peaks.compute_ceiling_for("fp16")
    assert ceil_fp16 == 174.78, f"fp16 ceiling {ceil_fp16} is not the max over measured paths"
    assert "Triton" in name_fp16, f"the winning path is not named in {name_fp16!r}"

    # fp32: cuBLAS is higher and STAYS the ceiling. Lowering the roof to Triton's figure would hide
    # the gap; the point is to report it, because a backend change is a real (if expensive) lever.
    # Asked literally -- a scalar kernel's ceiling IS fp32, and this pair holds the widest gap.
    ceil_fp32, _ = peaks.compute_ceiling_for("fp32")
    assert ceil_fp32 == 54.20, f"the fp32 roof was lowered to {ceil_fp32}, hiding the gap"

    # The gap is its own quantity: "at 84% of the roof" and "84% is all this backend reaches" imply
    # opposite actions -- keep tuning versus change code generator.
    reach = peaks.backend_reachable_frac("fp32")
    assert reach is not None and abs(reach - 45.61 / 54.20) < 1e-9, (
        f"backend_reachable_frac was {reach}, so the fp32 gap is invisible")

    # Cannot exceed 1.0, because the ceiling is the max of the pair.
    assert peaks.backend_reachable_frac("fp16") == 1.0, (
        "the fp16 reachable fraction exceeded 1.0, so the ceiling was not the max over paths")


def test_the_precorrection_ceiling_reports_an_impossible_fraction():
    """Reverse-verification: with the Triton figures absent, the defect must reappear.

    A test that passes both before and after a fix demonstrates nothing. This asserts the OLD
    behaviour is broken in the specific way the fix addresses.
    """
    pre = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.20, tf32_tflops=88.14, fp16_tflops=159.47)
    achieved = 165.0        # between cuBLAS's fp16 figure and Triton's, i.e. genuinely achievable

    assert achieved / pre.compute_ceiling_for("fp16")[0] > 1.0, (
        "the pre-fix ceiling did not produce an impossible fraction, so this test does not "
        "demonstrate the defect it exists for")
    assert achieved / DevicePeaks(**BOX1).compute_ceiling_for("fp16")[0] < 1.0, (
        "the fixed ceiling still reports an impossible fraction")

    # An unmeasured Triton ceiling must degrade to the old behaviour, not to a wrong number.
    assert pre.backend_reachable_frac("fp16") is None, (
        "an unmeasured Triton ceiling produced a reachable fraction, so a missing measurement "
        "became a claim")


def test_backend_unreachability_reaches_the_verdict_an_agent_reads():
    """A ceiling fix that never reaches the evidence changes no decision.

    Note the `sass` argument: the tf32 ceiling is only in force when the kernel's own instruction
    mix shows tensor cores, so a test that omits it is scored against fp32 and would assert the
    wrong roof. That mismatch was a real bug in the first version of this fix -- a tf32 reachable
    fraction printed beside an fp32 percentage, two roofs in one verdict.
    """
    peaks = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.20, tf32_tflops=88.14,
                        fp32_triton_tflops=45.61, tf32_triton_tflops=86.58)
    tc = {"instructions": 1000, "tensor_core": 120}
    v = classify(gpu_ms=1.0, cpu_issue_ms=None, flop_count=int(0.85 * 88.14 * 1e9),
                 byte_count=None, peaks=peaks, precision="tf32", sass=tc)

    assert v.evidence.get("compute_ceiling_used", "").startswith("tensor-core (tf32)"), (
        f"the tf32 ceiling is not in force, so this test asserts the wrong roof: "
        f"{v.evidence.get('compute_ceiling_used')!r}")
    assert "backend_reachable_frac" in v.evidence, (
        "the reachable fraction is absent from the evidence, so nothing an agent reads changed")
    basis = v.evidence.get("backend_reachable_basis", "")
    assert "Triton" in basis and "%" in basis, (
        f"the basis does not say what the fraction is or where it came from: {basis!r}")

    # The fraction must describe the ceiling actually used, not whatever precision was passed in.
    assert abs(v.evidence["backend_reachable_frac"] - round(86.58 / 88.14, 3)) < 1e-9, (
        "the reachable fraction does not match the ceiling the percentage is against")

    # A SCALAR kernel (no tensor cores) is scored against fp32, whose gap is the widest of the four
    # -- so this is the case that must not be skipped.
    scalar = classify(gpu_ms=1.0, cpu_issue_ms=None, flop_count=int(0.85 * 54.20 * 1e9),
                     byte_count=None, peaks=peaks, precision=None,
                     sass={"instructions": 1000, "tensor_core": 0})
    assert abs(scalar.evidence["backend_reachable_frac"] - round(45.61 / 54.20, 3)) < 1e-9, (
        "the fp32 backend gap -- the widest measured -- is not reported for a scalar kernel")

    # Nothing is reported when the backend reaches the roof, because there is then no gap to
    # describe. Note this is an EQUALITY test, not a tolerance: no cut-off exists on purpose. A
    # first version suppressed everything above 98% reachable, which hid the measured tf32 gap
    # (0.982) -- the precision most candidates actually compute in.
    agreeing = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.20, tf32_tflops=88.14,
                           tf32_triton_tflops=88.14)
    v2 = classify(gpu_ms=1.0, cpu_issue_ms=None, flop_count=int(0.85 * 88.14 * 1e9),
                  byte_count=None, peaks=agreeing, precision="tf32", sass=tc)
    assert "backend_reachable_frac" not in v2.evidence, (
        "a backend that reaches the roof was reported as limited by it")

    # ...and a small gap IS reported, which is the case the first threshold wrongly hid.
    narrow = DevicePeaks(dram_tbs=0.911, fp32_tflops=54.20, tf32_tflops=88.14,
                         tf32_triton_tflops=86.58)
    v3 = classify(gpu_ms=1.0, cpu_issue_ms=None, flop_count=int(0.85 * 88.14 * 1e9),
                  byte_count=None, peaks=narrow, precision="tf32", sass=tc)
    assert v3.evidence.get("backend_reachable_frac") == 0.982, (
        "the measured 98.2% tf32 gap was suppressed as too small, so the number is missing for "
        "the precision most candidates use")


def test_triton_ceiling_measurement_is_gated_on_correctness():
    """The throughput of a wrong kernel is not a ceiling.

    A kernel that skips work is faster, and that always looks like good news: this probe, before it
    had a correctness gate, reported Triton above cuBLAS on three of four precisions.
    """
    from kernel_optimizer.gpu import tritonmm

    assert set(tritonmm._TOL) == {"fp32", "tf32", "fp16", "bf16"}, (
        "a precision the calibration measures has no correctness tolerance, so its ceiling would "
        "be accepted unchecked")
    # fp32 must be pinned: the tl.dot default puts fp32 inputs on tf32 tensor cores, which measured
    # as an fp32 ratio of 1.584 -- "Triton is 58% faster than cuBLAS at fp32".
    assert tritonmm._IPREC["fp32"] == "ieee", (
        "fp32 is not pinned to ieee, so the fp32 ceiling is not measured in fp32")
    # Tolerances are input-rounding budgets, so they must order by precision.
    assert tritonmm._TOL["fp32"] < tritonmm._TOL["tf32"] < tritonmm._TOL["fp16"], (
        "the tolerance ordering does not follow precision, so one is not a rounding budget")

    src = inspect.getsource(tritonmm.reachable_tflops)
    assert "n_wrong" in src, "no wrong-result counter, so discarded configs would be invisible"


def test_calibration_carries_the_triton_ceilings_to_the_classifier():
    """A measurement that stops before the classifier changes nothing.

    Both links have silently broken before: a new field served as 0.0 forever because the schema
    version was not bumped, and a fix applied in one of three producers.
    """
    assert CALIBRATION_SCHEMA_VERSION >= 3, (
        "the schema version was not bumped for the Triton ceilings, so a cache written before them "
        "stays valid and every reachable fraction is served as 0.0 indefinitely")

    cal = Calibration(device_name="x", capability=[8, 9], sm_count=128, **BOX1)
    for name in ("fp32", "tf32", "fp16", "bf16"):
        assert getattr(cal, f"{name}_triton_tflops") > 0, f"{name} Triton ceiling did not persist"

    # The model default is 0.0, so an un-forwarded field reverts the ceiling to cuBLAS-only in
    # silence -- no exception, and every verdict still reads like a verdict.
    from kernel_optimizer.control import orchestrator as orch

    src = inspect.getsource(orch)
    for name in ("fp32", "tf32", "fp16", "bf16"):
        assert f"{name}_triton_tflops=self.calibration.{name}_triton_tflops" in src, (
            f"the orchestrator does not forward {name}_triton_tflops, so the classifier sees 0.0")


def test_agents_are_told_the_reachable_ceiling_not_only_the_library_one():
    """An agent told only the cuBLAS figure aims above what it can hit at fp32 and below at fp16."""
    from kernel_optimizer.agents.modules import _measured_ceilings_doc

    cal = Calibration(device_name="RTX 4090", capability=[8, 9], sm_count=128, **BOX1)
    doc = _measured_ceilings_doc(cal)

    assert "45.6" in doc, "the agents are not told the fp32 figure their backend can reach"
    assert "174.8" in doc, (
        "the agents are not told Triton beats cuBLAS at fp16 here, so they aim below their limit")
    # tf32 agrees within 2% and must NOT be listed: four near-identical lines bury the two that
    # matter, which is the failure mode a metric dump has.
    assert doc.count("from Triton") == 3, (
        f"expected only the precisions that differ, got {doc.count('from Triton')}")

    # No calibration must still render rather than assert a default.
    assert _measured_ceilings_doc(None) == ""
