"""G26 / J2-8 (N4): a candidate whose working set fits in L2 must not be reported DRAM-bound.

S2's own hard gate, and the reason it is a gate rather than a later refinement: S2's output is a
per-dimension list of verdicts, and when the criterion mislabels WHICH dimension binds, parallel
output just splits one error into two columns.

MEASURED, and the measurement is why one gate is not enough. `probe_binding_criterion_control.py`
builds five kernels whose limit is known by construction. The L2-resident streaming kernel (E):

    4090   reads 287%  of the DRAM roof  -> caught by _IMPOSSIBLE_FRAC = 1.05
    A800   reads 1.04  of the DRAM roof  -> SLIPS UNDER that threshold

Same kernel, same code, different card: the A800's L2 is 40 MiB against the 4090's 72 MiB, so the
inflation is smaller and lands in the possible-looking range. A fixed threshold provably cannot
catch this case, which is what makes the working-set test necessary rather than belt-and-braces.

These tests drive the real `classify()`.
"""
from __future__ import annotations

from kernel_optimizer.evaluation.bottleneck import DevicePeaks, classify

_MIB = 2 ** 20


def _peaks(l2_mib: int = 40) -> DevicePeaks:
    """A800-like ceilings: the card on which the threshold alone fails."""
    return DevicePeaks(dram_tbs=1.7025, fp32_tflops=18.7, tf32_tflops=111.8,
                       fp16_tflops=229.0, bf16_tflops=234.0, l2_bytes=l2_mib * _MIB)


def test_an_l2_resident_kernel_is_not_reported_dram_bound():
    """N4, the must-fail test: the A800 case that slips under the impossible-fraction threshold."""
    # 8 MiB working set on a card with 40 MiB of L2, at a fraction that looks entirely possible.
    v = classify(gpu_ms=0.004977, cpu_issue_ms=None, flop_count=1_000,
                 byte_count=8 * _MIB, peaks=_peaks(), precision="fp32")
    assert v.kind != "memory_bound", (
        "a kernel whose whole working set fits in L2 was reported DRAM-bound. Its logical bytes "
        "never crossed the memory bus, so the bandwidth reading is not a measurement of this "
        "kernel -- and on the control that motivated this, the real limit was occupancy while the "
        "bandwidth reading was above 100%%. Verdict was: %s" % v.kind)
    assert v.kind == "unknown", v.kind
    ev = v.evidence
    assert ev.get("dram_applicable") is False, (
        "the dram dimension was not marked inapplicable, so a per-dimension consumer would still "
        "treat this reading as a real bandwidth verdict: %s" % ev)
    # The numbers that decided it must be reported, so a reader can check the call.
    assert ev.get("working_set_mib") == 8.0, ev
    assert ev.get("l2_mib") == 40.0, ev
    assert "fits in" in (ev.get("dram_inapplicable_reason") or ""), ev


def test_the_threshold_alone_would_have_missed_it():
    """The reverse control: with the L2 size unmeasured, this kernel IS mislabelled.

    Without it the test above could pass for the wrong reason -- some other branch catching the
    kernel -- and we would not know the working-set gate did any work. `l2_bytes=0` means
    unmeasured, which is exactly the pre-fix state.
    """
    v = classify(gpu_ms=0.004977, cpu_issue_ms=None, flop_count=1_000,
                 byte_count=8 * _MIB, peaks=_peaks(l2_mib=0), precision="fp32")
    assert v.kind == "memory_bound", (
        "with L2 unmeasured this kernel should still read DRAM-bound -- if it does not, the test "
        "above is passing for some other reason and proves nothing about the working-set gate")


def test_an_unmeasured_l2_never_invalidates_a_real_bandwidth_verdict():
    """Direction of the failure: `l2_bytes == 0` must not silence a genuine memory-bound verdict.

    An unmeasured L2 declaring the dimension inapplicable would suppress the true verdict on every
    box that failed to measure it -- worse than the bug being fixed.
    """
    v = classify(gpu_ms=0.840491, cpu_issue_ms=None, flop_count=10**9,
                 byte_count=1351 * _MIB, peaks=_peaks(l2_mib=0), precision="fp32")
    assert v.kind == "memory_bound", v.kind
    assert v.evidence.get("dram_applicable") is not False


def test_a_real_dram_bound_kernel_still_reports_memory_bound():
    """L3:48's shape: 1.351 GB working set, far above any L2 here. Must be unaffected."""
    v = classify(gpu_ms=0.840491, cpu_issue_ms=None, flop_count=10**9,
                 byte_count=1351 * _MIB, peaks=_peaks(), precision="fp32")
    assert v.kind == "memory_bound", (
        "the working-set gate suppressed a genuine memory-bound verdict on a 1.351 GB working set "
        "against 40 MiB of L2 -- that is 34x larger, so it cannot be L2-resident: %s" % v.kind)
    assert v.evidence.get("dram_applicable") is not False


def test_the_boundary_is_the_l2_size_not_a_constant():
    """The gate must scale with the box's measured L2, since the two cards differ by 1.8x."""
    # 50 MiB: inside a 72 MiB L2 (4090-like), outside a 40 MiB one (A800-like).
    on_4090 = classify(gpu_ms=0.031106, cpu_issue_ms=None, flop_count=1000,
                       byte_count=50 * _MIB, peaks=_peaks(l2_mib=72), precision="fp32")
    on_a800 = classify(gpu_ms=0.031106, cpu_issue_ms=None, flop_count=1000,
                       byte_count=50 * _MIB, peaks=_peaks(l2_mib=40), precision="fp32")
    assert on_4090.evidence.get("dram_applicable") is False, (
        "50 MiB fits in a 72 MiB L2 and must be flagged there")
    assert on_a800.kind == "memory_bound", (
        "50 MiB does NOT fit in a 40 MiB L2, so the same kernel must keep its bandwidth verdict "
        "on that card -- a gate that fires on both cards is a hardcoded constant, not a "
        "per-box measurement")


def test_the_impossible_fraction_path_still_works():
    """The 4090's 287% case must still be caught, by the threshold, and say so."""
    v = classify(gpu_ms=0.205, cpu_issue_ms=None, flop_count=1000,
                 byte_count=1000 * _MIB, peaks=_peaks(l2_mib=40), precision="fp32")
    assert v.kind == "unknown", v.kind
    assert v.evidence.get("impossible_dram_fraction", 0) > 105, v.evidence
    # This one is NOT the working-set case, and must not claim to be.
    assert v.evidence.get("dram_applicable") is not False, (
        "a kernel caught by the impossible-fraction threshold was labelled as L2-resident, which "
        "is a different diagnosis and sends the reader after the wrong cause")
